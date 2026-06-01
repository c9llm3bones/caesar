"""This module implements the protein structure autoencoders from the manuscript."""

from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk

# alphafold dependencies
from salad.aflib.model.geometry import Vec3Array, Rot3Array
from salad.aflib.model.all_atom_multimer import (
    get_atom14_mask, atom37_to_atom14, make_transform_from_reference)
from salad.modules.utils.alphafold_loss import violation_loss

# basic module imports
from salad.modules.basic import (
    Linear, MLP, init_relu,
    init_zeros, init_linear,
    block_stack
)

# import geometry utils
from salad.modules.utils.geometry import (
    index_mean, index_sum, extract_aa_frames,
    extract_neighbours, distance_rbf,
    unique_chain, positions_to_ncacocb, index_align,
    single_protein_sidechains, compute_pseudo_cb,
    get_random_neighbours, get_spatial_neighbours,
    get_neighbours, axis_index, distance_one_hot)
from salad.modules.utils.dssp import assign_dssp

# sparse geometric module imports
from salad.modules.geometric import (
    SparseStructureAttention, SparseAttention, SparseSemiEquivariantPointAttention,
    sequence_relative_position,
    distance_features, direction_features, pair_vector_features,
    position_rotation_features
)


def extract_aa_frames_atom37(positions: Vec3Array):
    """Extract residue frames from N/CA/C and local positions for all atom37 atoms."""
    frames = make_transform_from_reference(
        a_xyz=positions[..., 0],
        b_xyz=positions[..., 1],
        c_xyz=positions[..., 2])
    local_positions = frames[..., None].apply_inverse_to_point(positions)
    return frames, local_positions


def atom37_to_ncacocb(pos37: jnp.ndarray):
    """Construct N, CA, C, O, pseudo-CB view from atom37 coordinates."""
    pseudo_cb = compute_pseudo_cb(pos37)
    return jnp.concatenate(
        (pos37[:, :3], pos37[:, 4:5], pseudo_cb[:, None, :]),
        axis=-2)

class StructureAutoencoder(hk.Module):
    """Wrapper class for protein structure autoencoder training"""
    def __init__(self, config,
                 name: Optional[str] = "structure_autoencoder"):
        super().__init__(name)
        self.config = config

    def prepare_data(self, data):
        """Prepare model inputs from a batch of data."""
        c = self.config
        pos = data["all_atom_positions"]
        atom_mask = data["all_atom_mask"]
        aatype = data["aa_gt"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        no_random = jnp.asarray(data.get("no_random", False))
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        if atom37_parallel_mode not in ("none", "parallel"):
            raise ValueError(
                f"Unknown atom37 parallel mode: {atom37_parallel_mode}")

        # recast positions into atom14:
        # keep a parallel canonical atom37 branch when available, but keep the
        # main trunk working with atom14 -> NCACOCB as before.
        atom_pos_37 = atom_mask_37 = None
        if pos.shape[-2] == 37:
            atom_pos_37 = pos
            atom_mask_37 = atom_mask
            pos, atom_mask = atom37_to_atom14(aatype, pos, atom_mask)
        else:
            pos = pos[:, :14]
            atom_mask = atom_mask[:, :14]

        # uniquify chain IDs across batches:
        chain = unique_chain(chain, batch)
        mask = data["seq_mask"] * data["residue_mask"] * atom_mask[:, :3].all(axis=-1)
        
        # subtract the center from all positions
        center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
        pos = pos - center[:, None]
        if atom_pos_37 is not None:
            atom_pos_37 = atom_pos_37 - center[:, None]
        # set the positions of all masked atoms to the pseudo Cb position
        pos = jnp.where(
            atom_mask[..., None], pos,
            compute_pseudo_cb(pos)[:, None, :])
        pos_14 = pos
        atom_mask_14 = atom_mask
        pos37_gt = pos37_init = None
        if atom_pos_37 is not None:
            # Replace masked atom37 coordinates with pseudo-CB to avoid
            # undefined coordinates leaking into the geometric pipeline.
            atom_pos_37 = jnp.where(
                atom_mask_37[..., None], atom_pos_37,
                compute_pseudo_cb(atom_pos_37)[:, None, :])
            pos37_gt = atom_pos_37
            pos37_init = jax.random.normal(hk.next_rng_key(), pos37_gt.shape)
            pos37_init = jnp.where(no_random, pos37_gt, pos37_init)

        # get ncacocb positions
        pos_ncacocb = positions_to_ncacocb(pos)

        # make distance map
        cb = Vec3Array.from_array(pos_ncacocb[:, -1])
        # add noise
        noise_scale = jnp.where(no_random, 0.0, 0.3)
        cb += Vec3Array.from_array(noise_scale * jax.random.normal(hk.next_rng_key(), list(cb.shape) + [3]))
        dmap = (cb[:, None] - cb[None, :]).norm()
        dmap_mask = batch[:, None] == batch[None, :]

        # set all-atom-position target
        atom_pos = pos_14
        atom_mask = atom_mask_14

        # assign dssp
        dssp, _, _ = assign_dssp(atom_pos, batch, mask)

        # set initial backbone positions
        pos = jax.random.normal(hk.next_rng_key(), pos_ncacocb.shape)
        pos = jnp.where(no_random, pos_ncacocb, pos)
        result = dict(pos=pos, pos_gt=pos_ncacocb, pos_input=pos_ncacocb,
                      dssp=dssp, dmap=dmap, dmap_mask=dmap_mask,
                      chain_index=chain, mask=mask,
                      atom_pos=atom_pos, atom_mask=atom_mask,
                      all_atom_positions=pos_14,
                      all_atom_mask=atom_mask_14)
        if atom_pos_37 is not None:
            result.update(
                all_atom_positions_37=atom_pos_37,
                all_atom_mask_37=atom_mask_37)
        if atom37_parallel_mode == "parallel":
            if pos37_gt is None:
                raise ValueError(
                    "atom37_parallel_mode='parallel' requires atom37 input "
                    "coordinates in the dataloader.")
            result.update(
                pos37=pos37_init,
                pos37_gt=pos37_gt,
                pos37_input=pos37_gt)
        return result

    def __call__(self, data):
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        encoder = Encoder(c)
        decoder = Decoder(c)
        # optionally use VQ
        if c.codebook_size:
            vq = VQState if c.state else VQ
            mapped_axes = []
            if c.rebatch:
                mapped_axes.append("rebatch_ax")
            if c.multigpu:
                mapped_axes.append("batch_ax")
            quantize = vq(c.codebook_size, c.affine, mapped_axes=mapped_axes)
        # or FSQ (not used in the manuscript)
        if c.fsq:
            fsq = FSQ()

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        trace = {}
        trace["prep/mask_sum"] = jnp.sum(data["mask"]).astype(jnp.float32)
        trace["prep/pos_gt_mean"] = jnp.mean(data["pos_gt"])
        trace["prep/pos_init_mean"] = jnp.mean(data["pos"])
        trace["prep/dmap_mean"] = jnp.mean(data["dmap"])
        # optionally apply noise to the inputs
        if c.input_diffusion:
            clean_latent = encoder(data)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data))
        latent = encoder(data)
        trace["enc/latent_mean"] = jnp.mean(latent)
        trace["enc/latent_std"] = jnp.std(latent)
        trace["enc/latent_slice"] = latent[:2, :5]
        # optionally apply noise to the latents (not used in the manuscript)
        if c.latent_diffusion:
            # NOTE: constraining latents is necessary for diffusion to work
            # with a trainable encoder. Otherwise, the model learns to cheat.
            # we do this by applying a parameter-less LayerNorm to the latent
            # vectors. This fixes the variance of the latent vectors to 1 and
            # bounds the achievable signal-to-noise ratio during the diffusion
            # process. Thus, the model has to actively learn to denoise.
            latent = hk.LayerNorm([-1], False, False)(latent)
            data["clean_latent"] = latent
            latent, time = self.prepare_latent_diffusion(latent, data)
            if not c.vp_diffusion:
                data["skip_latent"] = latent / jnp.maximum(1 + time[:, None] ** 2, 1e-3)
                latent = latent / jnp.maximum(jnp.sqrt(1 + time[:, None] ** 2), 1e-3)
            data["time"] = time
        # optionally apply VQ or FSQ
        if c.codebook_size:
            if c.state:
                latent, codebook_index, codebook_losses, state_update = quantize(latent, data["mask"])
            else:
                latent, codebook_index, codebook_losses = quantize(latent, data["mask"])
            data["codebook_index"] = codebook_index
        if c.fsq:
            latent, _ = fsq(latent)
        data["latent"] = latent

        # decoder recycling
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            if atom37_parallel_mode == "parallel":
                prev.update(
                    pos37=result["pos37"],
                    local37=result["local37"])
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        if atom37_parallel_mode == "parallel":
            prev.update(
                pos37=data["pos37"],
                local37=jnp.zeros((data["pos37"].shape[0], c.local_size), dtype=jnp.float32))
        if not hk.running_init():
            fixed_recycle = data.get("fixed_recycle", None)
            if fixed_recycle is not None:
                count = fixed_recycle
            elif c.eval:
                count = c.num_recycle
            else:
                count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        prep0 = decoder.prepare_features(data, prev)
        if atom37_parallel_mode == "parallel":
            local0, pos0, local37_0, pos37_0, resi0, chain0, batch0, mask0 = prep0
            trace["dec/local37_0_mean"] = jnp.mean(local37_0)
            trace["dec/local37_0_std"] = jnp.std(local37_0)
            trace["dec/pos37_0_mean"] = jnp.mean(pos37_0)
        else:
            local0, pos0, resi0, chain0, batch0, mask0 = prep0
        trace["dec/local0_mean"] = jnp.mean(local0)
        trace["dec/local0_std"] = jnp.std(local0)
        result = decoder(data, prev)
        if c.codebook_size:
            result["codebook_losses"] = codebook_losses
        total, losses = decoder.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        if c.codebook_size and c.state:
            out_dict["_state_update"] = state_update
        
        return total, out_dict

    def add_noise(self, latent, batch):
        """Add noise to latents.
        
        Not used in the manuscript.
        """
        noise_level = jax.random.normal(hk.next_rng_key(), batch.shape)[batch]
        noise_level = jnp.exp(noise_level)
        noise = jax.random.normal(hk.next_rng_key(), latent.shape) * noise_level
        return latent + noise

    def prepare_input_diffusion(self, data):
        """When training as a structure diffusion model, prepare model input.
        
        Not used in the manuscript.
        """
        c = self.config
        batch = data["batch_index"]
        pos = data["pos_input"]
        pos -= index_mean(pos[:, 1], batch, data["mask"][:, None])[:, None] # FIXME
        time = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
        if "time" in data:
            time = data["time"] * jnp.ones_like(time)
        s = 0.01
        time = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) / jnp.cos(s / (1 + s) * jnp.pi / 2)
        time = jnp.sqrt(jnp.clip(1 - time ** 2, 0, 1))
        noise = c.sigma_data * jax.random.normal(hk.next_rng_key(), pos.shape)
        pos = jnp.sqrt(1 - time[:, None, None] ** 2) * pos + time[:, None, None] * noise
        return dict(pos_input=pos, time=time)

    def prepare_latent_diffusion(self, latent, data):
        """When training as a latent diffusion model, prepare model input.
        
        Not used in the manuscript.
        """
        c = self.config
        batch = data["batch_index"]
        noise = jax.random.normal(hk.next_rng_key(), latent.shape)
        if c.vp_diffusion:
            time = jax.random.uniform(hk.next_rng_key(), batch.shape)[batch]
        else:
            time = jnp.exp(1.0 + 1.2 * jax.random.normal(hk.next_rng_key(), batch.shape)[batch])
        # FIXME: allow time input
        if "time" in data:
            time = data["time"] * jnp.ones_like(time)
        if c.vp_diffusion:
            s = 0.01
            time = jnp.cos((time + s) / (1 + s) * jnp.pi / 2) / jnp.cos(s / (1 + s) * jnp.pi / 2)
            time = jnp.sqrt(jnp.clip(1 - time ** 2, 0, 1))
            latent = jnp.sqrt(1 - time[:, None] ** 2) * latent + time[:, None] * noise * 5.0
        else:
            latent = latent + time[:, None] * noise
        return latent, time

class StructureDecoder(StructureAutoencoder):
    """Wrapper class for training a structure autoencoder with a fixed encoder.
    
    Not used in the manuscript.
    """
    def __init__(self, config,
                 name: Optional[str] = "structure_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        # fixed encoder
        encoder = assign_state(c, c.param_path)
        # learnable decoder
        decoder = Decoder(c)

        data.update(self.prepare_data(data))
        latent, codebook_index = encoder(hk.next_rng_key(), data)
        latent = jax.lax.stop_gradient(latent)
        data["latent"] = latent

        # decoder recycling
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            if atom37_parallel_mode == "parallel":
                prev.update(
                    pos37=result["pos37"],
                    local37=result["local37"])
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        if atom37_parallel_mode == "parallel":
            prev.update(
                pos37=data["pos37"],
                local37=jnp.zeros((data["pos37"].shape[0], c.local_size), dtype=jnp.float32))
        if not hk.running_init():
            fixed_recycle = data.get("fixed_recycle", None)
            if fixed_recycle is not None:
                count = fixed_recycle
            elif c.eval:
                count = c.num_recycle
            else:
                count = jax.random.randint(hk.next_rng_key(), (), 0, 4)
            prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)
        total, losses = decoder.loss(data, result)
        out_dict = dict(
            results=result,
            losses=losses
        )
        return total, out_dict

    def add_noise(self, latent, batch):
        """Add noise to latents."""
        noise_level = jax.random.normal(hk.next_rng_key(), batch.shape)[batch]
        noise_level = jnp.exp(noise_level)
        noise = jax.random.normal(hk.next_rng_key(), latent.shape) * noise_level
        return latent + noise

class StructureAutoencoderInference(StructureAutoencoder):
    """Wrapper class for autoencoder evaluation."""
    def __call__(self, data):
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        encoder = Encoder(c)
        decoder = Decoder(c)
        if c.codebook_size:
            vq = VQState if c.state else VQ
            quantize = vq(c.codebook_size)

        # convert coordinates, center & add encoded features
        data.update(self.prepare_data(data))
        trace = {}
        if getattr(c, "trace", False):
            trace["prep/pos"] = data["pos"]
            trace["prep/pos_gt"] = data["pos_gt"]
            trace["prep/mask"] = data["mask"]
            trace["prep/dmap"] = data["dmap"]
            # grab encoder-side init features
            enc_prep = encoder.prepare_features(data)
            if len(enc_prep) == 8:
                enc_local0, _, _, _, _, _, enc_neigh0, enc_lfeat0 = enc_prep
                trace["enc/local0"] = enc_local0
                trace["enc/neighbours0"] = enc_neigh0
                trace["enc/local_features0"] = enc_lfeat0
            elif len(enc_prep) == 12:
                (enc_local0, _, enc_local37_0, _,
                 _, _, _, _, enc_neigh0, enc_lfeat0,
                 enc_neigh37_0, enc_lfeat37_0) = enc_prep
                trace["enc/local0"] = enc_local0
                trace["enc/neighbours0"] = enc_neigh0
                trace["enc/local_features0"] = enc_lfeat0
                trace["enc/local37_0"] = enc_local37_0
                trace["enc/neighbours37_0"] = enc_neigh37_0
                trace["enc/local37_features0"] = enc_lfeat37_0
        # optionally apply noise to the inputs
        if c.input_diffusion:
            clean_latent = encoder(data)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data))
        latent = encoder(data)
        if getattr(c, "trace", False):
            trace["enc/latent"] = latent
        # optionally apply noise to the latents
        if c.latent_diffusion:
            if "latent" in data:
                latent = data["latent"]
            # FIXME
            # latent = hk.LayerNorm([-1], False, False)(latent)
            data["clean_latent"] = latent
            latent, time = self.prepare_latent_diffusion(latent, data)
            if not c.vp_diffusion:
                data["skip_latent"] = latent / jnp.maximum(1 + time[:, None] ** 2, 1e-3)
                latent = latent / jnp.maximum(jnp.sqrt(1 + time[:, None] ** 2), 1e-3)
            data["time"] = time
            jax.debug.print("time {time}", time=data["time"][0])
        if c.codebook_size:
            latent, codebook_index, _ = quantize(latent, data["mask"])
        data["latent"] = latent

        # decoder recycling
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            if atom37_parallel_mode == "parallel":
                prev.update(
                    pos37=result["pos37"],
                    local37=result["local37"])
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        if atom37_parallel_mode == "parallel":
            prev.update(
                pos37=data["pos37"],
                local37=jnp.zeros((data["pos37"].shape[0], c.local_size), dtype=jnp.float32))

        count = c.num_recycle
        prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)
        if getattr(c, "trace", False):
            trace["dec/pos"] = result["pos"]
            trace["dec/local"] = result["local"]
            if atom37_parallel_mode == "parallel":
                trace["dec/pos37"] = result["pos37"]
                trace["dec/local37"] = result["local37"]
            trace["dec/atom_pos"] = result["atom_pos"]
            trace["dec/aa"] = result["aa"]

        mask = data["mask"]
        # compute decoder perplexity and sequence recovery
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(mask.sum(), 1)
        perplexity = jnp.exp(aa_nll)
        aatype = jnp.argmax(result["aa"], axis=-1)
        recovery = ((data["aa_gt"] == aatype) * mask).sum() / mask.sum()
        # compute reconstruction RMSD, LDDT and TM score
        pos_gt = data["pos_gt"]
        pos = result["pos"]
        pos_gt = index_align(pos_gt, pos, data["batch_index"], mask)
        di2 = ((pos[:, 1] - pos_gt[:, 1]) ** 2).sum(axis=-1)
        rmsd_ca = jnp.sqrt((di2 * mask).sum() / mask.sum())
        d02 = (1.24 * (mask.sum() - 15) ** (1 / 3) - 1.8) ** 2
        inner = 1 / (1 + di2 / d02) * mask
        tm = inner.sum() / mask.sum()
        dca_gt = jnp.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], axis=-1)
        dca = jnp.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
        pair_mask = mask[:, None] * mask[None, :]
        derr = abs(dca_gt - dca) * pair_mask
        threshold = jnp.array([0.5, 1.0, 2.0, 4.0])
        Rinc = 15.0
        pair_mask *= dca_gt < Rinc
        in_threshold = (derr[..., None] < threshold) * pair_mask[..., None]
        lddt_ca = (in_threshold.sum(axis=1) / jnp.maximum(pair_mask[..., None].sum(axis=1), 1)).mean(axis=-1)
        lddt_ca = jnp.where(mask, lddt_ca, 0)
        # compute AlphaFold structural violation loss
        res_mask = data["mask"]
        pred_mask = get_atom14_mask(aatype) * res_mask[:, None]
        violation, _ = violation_loss(aatype,
                                      data["residue_index"],
                                      result["atom_pos"],
                                      pred_mask,
                                      res_mask,
                                      clash_overlap_tolerance=1.5,
                                      violation_tolerance_factor=2.0,
                                      chain_index=data["chain_index"],
                                      batch_index=data["batch_index"],
                                      per_residue=False)
        violation_error = violation.mean()

        latent = data["latent"]
        if "predicted_latent" in result:
            print("returning predicted latent")
            latent = result["predicted_latent"]
        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=latent,
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            tm=tm,
            lddt=lddt_ca,
            time=data["time"],
            dssp=data["dssp"],
            violation=violation_error)
        if atom37_parallel_mode == "parallel":
            out["pos37"] = result["pos37"]
            out["trajectory37"] = result["trajectory37"]
        if c.codebook_size:
            out["codebook_index"] = codebook_index
        if getattr(c, "trace", False):
            out["__trace"] = trace
        return out
    
class StructureDecoderInference(StructureDecoder):
    """Wrapper class for StructureDecoder evaluation."""
    def __call__(self, data):
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        encoder = assign_state(c, c.param_path)
        decoder = Decoder(c)

        #data.update(self.prepare_data(data))
        latent, codebook_index = encoder(hk.next_rng_key(), data)
        latent = jax.lax.stop_gradient(latent)
        data["latent"] = latent

        # decoder recycling
        def iteration_body(i, prev):
            result = decoder(data, prev)
            prev = dict(
                pos=result["pos"],
                local=result["local"]
            )
            if atom37_parallel_mode == "parallel":
                prev.update(
                    pos37=result["pos37"],
                    local37=result["local37"])
            return jax.lax.stop_gradient(prev)
        prev = dict(
            pos=data["pos"],
            local=jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        if atom37_parallel_mode == "parallel":
            prev.update(
                pos37=data["pos37"],
                local37=jnp.zeros((data["pos37"].shape[0], c.local_size), dtype=jnp.float32))

        count = c.num_recycle
        prev = jax.lax.stop_gradient(hk.fori_loop(0, count, iteration_body, prev))
        result = decoder(data, prev)

        # compute amino acid prediction perplexity, sequence recovery
        mask = data["mask"]
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(mask.sum(), 1)
        perplexity = jnp.exp(aa_nll)
        aatype = jnp.argmax(result["aa"], axis=-1)
        recovery = ((data["aa_gt"] == aatype) * mask).sum() / mask.sum()
        # compute reconstruction RMSD, LDDT and TM score
        pos_gt = data["pos_gt"]
        pos = result["pos"]
        pos_gt = index_align(pos_gt, pos, data["batch_index"], mask)
        di2 = ((pos[:, 1] - pos_gt[:, 1]) ** 2).sum(axis=-1)
        rmsd_ca = jnp.sqrt((di2 * mask).sum() / mask.sum())
        d02 = (1.24 * (mask.sum() - 15) ** (1 / 3) - 1.8) ** 2
        inner = 1 / (1 + di2 / d02) * mask
        tm = inner.sum() / mask.sum()
        dca_gt = jnp.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], axis=-1)
        dca = jnp.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
        pair_mask = mask[:, None] * mask[None, :]
        derr = abs(dca_gt - dca) * pair_mask
        threshold = jnp.array([0.5, 1.0, 2.0, 4.0])
        Rinc = 15.0
        pair_mask *= dca_gt < Rinc
        in_threshold = (derr[..., None] < threshold) * pair_mask[..., None]
        lddt_ca = (in_threshold.sum(axis=1) / jnp.maximum(pair_mask[..., None].sum(axis=1), 1)).mean(axis=-1)
        lddt_ca = jnp.where(mask, lddt_ca, 0)

        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=data["latent"],
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            tm=tm,
            lddt=lddt_ca,
            dssp=data["dssp"])
        if c.codebook_size:
            out["codebook_index"] = codebook_index
        return out
    
class AssignState(StructureAutoencoder):
    """Fixed encoder wrapper."""
    def __call__(self, data):
        c = self.config
        encoder = Encoder(c)
        if c.codebook_size:
            vq = VQState if c.state else VQ
            quantize = vq(c.codebook_size)
        data.update(self.prepare_data(data))
        latent = encoder(data)
        if c.codebook_size:
            latent, codebook_index, _ = quantize(latent, data["mask"])
            return latent, codebook_index
        return latent, None

def assign_state(config, param_path):
    """Construct a fixed encoder from a model configuration and parameter path."""
    import pickle
    apply = hk.transform(lambda x: AssignState(config)(x)).apply
    with open(param_path, "rb") as f:
        params = pickle.load(f)
    def inner(key, data):
        return apply(params, key, data)
    return inner

class AADecoderBlock(hk.Module):
    """Amino acid sequence decodder block."""
    def __init__(self, config, name: Optional[str] = "aa_decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, features, pos,
                 neighbours, resi, chain, batch, mask):
        c = self.config
        # embed pair features
        pair, pair_mask = aa_decoder_pair_features(c)(
            Vec3Array.from_array(pos), neighbours, resi, chain, batch, mask)
        pair += hk.LayerNorm([-1], True, True)(
            Linear(pair.shape[-1], bias=False)(
                jax.nn.one_hot(aa, 21)))[neighbours]
        pair = MLP(
            2 * c.pair_size, c.pair_size, activation=jax.nn.gelu,
            final_init=init_linear())(pair)
        # attention / message passing module
        features += SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        # local feature transition (always enabled)
        features += EncoderUpdate(c)(
                hk.LayerNorm([-1], True, True)(features), pos, chain, batch, mask)
        return features

class AADecoderStack(hk.Module):
    """Amino acid sequence decoder stack."""
    def __init__(self, config, depth=None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, aa, local, pos, neighbours,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(aa, data, pos,
                                neighbours,
                                resi, chain,
                                batch, mask)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(AADecoderBlock)))
        local = hk.LayerNorm([-1], True, True)(stack(local))
        return local

class EncoderBlock(hk.Module):
    """Encoder block."""
    def __init__(self, config, name: Optional[str] = "encoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 resi, chain, batch, mask):
        c = self.config
        # embed pair features
        nr = int(getattr(c, "num_random_neighbours", 32))
        neighbours = extract_neighbours(16, 16, nr)(
            Vec3Array.from_array(pos),
            resi, chain, batch, mask)
        pair, pair_mask = aa_decoder_pair_features(c)(
            Vec3Array.from_array(pos), neighbours, resi, chain, batch, mask)
        pair = MLP(
            2 * c.pair_size, c.pair_size, activation=jax.nn.gelu,
            final_init=init_linear())(pair)
        # attention / message passing module
        features += SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        # local feature transition (always enabled)
        features += EncoderUpdate(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos, chain, batch, mask)
        return features


class EncoderBlockAtom37(hk.Module):
    """Encoder block for the parallel atom37 branch."""
    def __init__(self, config, name: Optional[str] = "encoder_block_atom37"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos37,
                 resi, chain, batch, mask):
        c = self.config
        nr = int(getattr(c, "num_random_neighbours", 32))
        neighbours = extract_neighbours(16, 16, nr)(
            Vec3Array.from_array(pos37),
            resi, chain, batch, mask)
        pair, pair_mask = aa_decoder_pair_features_atom37(c)(
            Vec3Array.from_array(pos37), neighbours, resi, chain, batch, mask)
        pair = MLP(
            2 * c.pair_size, c.pair_size, activation=jax.nn.gelu,
            final_init=init_linear())(pair)
        features += SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos37, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        features += EncoderUpdateAtom37(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos37, chain, batch, mask)
        return features


class EncoderStack(hk.Module):
    """Encoder stack."""
    def __init__(self, config, depth=None, name: Optional[str] = "encoder_stack"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, local, pos,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(data, pos,
                                resi, chain,
                                batch, mask)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(EncoderBlock)))
        local = hk.LayerNorm([-1], True, True)(stack(local))
        return local


class EncoderStackAtom37(hk.Module):
    """Encoder stack for the parallel atom37 branch."""
    def __init__(self, config, depth=None, name: Optional[str] = "encoder_stack_atom37"):
        super().__init__(name)
        self.config = config
        self.depth = depth or 3

    def __call__(self, local, pos37,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                data = block(c)(data, pos37,
                                resi, chain,
                                batch, mask)
                return data
            return _inner
        stack = block_stack(
            self.depth, block_size=1, with_state=False)(
                hk.remat(stack_inner(EncoderBlockAtom37)))
        local = hk.LayerNorm([-1], True, True)(stack(local))
        return local


class Encoder(hk.Module):
    """Equivariant protein structure encoder."""
    def __init__(self, config, name: Optional[str] = "encoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, data):
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        prep = self.prepare_features(data)
        if atom37_parallel_mode == "parallel":
            if getattr(c, "trace", False):
                (local, pos, local37, pos37, resi, chain, batch, mask,
                 neighbours, local_features, neighbours37, local_features37) = prep
            else:
                local, pos, local37, pos37, resi, chain, batch, mask = prep
            local = EncoderStack(c, depth=c.encoder_depth)(
                local, pos, resi, chain, batch, mask)
            local37 = EncoderStackAtom37(c, depth=c.encoder_depth)(
                local37, pos37, resi, chain, batch, mask)
            fused = jnp.concatenate((local, local37), axis=-1)
            fused = MLP(
                4 * c.local_size, c.local_size,
                activation=jax.nn.gelu,
                bias=False,
                final_init=init_linear(),
                name="atom37_latent_fusion")(fused)
            fused = hk.LayerNorm([-1], True, True)(fused)
            if c.noembed:
                return fused
            return Linear(c.latent_size, initializer=init_linear(), bias=False)(fused)
        if getattr(c, "trace", False):
            local, pos, resi, chain, batch, mask, neighbours, local_features = prep
        else:
            local, pos, resi, chain, batch, mask = prep
        local = EncoderStack(c, depth=c.encoder_depth)(
            local, pos, resi, chain, batch, mask)
        if c.noembed:
            return local
        local = hk.LayerNorm([-1], True, True)(local)
        return Linear(c.latent_size, initializer=init_linear(), bias=False)(local)
    
    def prepare_features(self, data):
        """Construct encoder input features from a batch of data."""
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        atom37_encoder_mode = getattr(c, "atom37_encoder_mode", "none")
        pos = data["pos_input"]
        if c.noise_encoder and not c.eval:
            pos += c.noise_encoder * jax.random.normal(hk.next_rng_key(), shape=pos.shape)
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        pos = Vec3Array.from_array(pos)
        neighbours = extract_neighbours(5, 5, 0)(
            pos, resi, chain, batch, mask)
        local_features = init_local_features(c)(
            pos, neighbours, resi, chain, batch, mask)
        if atom37_parallel_mode == "none" and atom37_encoder_mode != "none":
            if "all_atom_positions_37" not in data or "all_atom_mask_37" not in data:
                raise ValueError(
                    f"Encoder atom37 mode {atom37_encoder_mode!r} requires "
                    "`all_atom_positions_37` and `all_atom_mask_37` in the batch.")
            atom37_pos = data["all_atom_positions_37"] / c.sigma_data
            atom37_mask = data["all_atom_mask_37"].astype(atom37_pos.dtype)
            atom37_pos = atom37_pos * atom37_mask[..., None]
            if atom37_encoder_mode == "abs_concat":
                atom37_features = jnp.concatenate((
                    atom37_pos.reshape(atom37_pos.shape[0], -1),
                    atom37_mask), axis=-1)
                local_features = jnp.concatenate(
                    (local_features, atom37_features), axis=-1)
            else:
                raise ValueError(
                    f"Unknown atom37 encoder mode: {atom37_encoder_mode}")
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        if c.time_embedding and c.input_diffusion and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 1, bins=100)
            local += Linear(
                local.shape[-1], bias=False, initializer="linear")(time)
        local = hk.LayerNorm([-1], True, True)(local)

        if atom37_parallel_mode == "parallel":
            if "pos37_input" not in data:
                raise ValueError(
                    "atom37_parallel_mode='parallel' requires `pos37_input` in data.")
            pos37 = data["pos37_input"]
            if c.noise_encoder and not c.eval:
                pos37 += c.noise_encoder * jax.random.normal(
                    hk.next_rng_key(), shape=pos37.shape)
            pos37 = Vec3Array.from_array(pos37)
            neighbours37 = extract_neighbours(5, 5, 0)(
                pos37, resi, chain, batch, mask)
            local_features37 = init_local_features_atom37(c)(
                pos37, neighbours37, resi, chain, batch, mask)
            local37 = MLP(
                4 * c.local_size, c.local_size,
                activation=jax.nn.gelu,
                bias=False,
                final_init=init_linear(),
                name="atom37_encoder_input")(local_features37)
            if c.time_embedding and c.input_diffusion and "time" in data:
                time = data["time"]
                time = distance_rbf(time, 0, 1, bins=100)
                local37 += Linear(
                    local37.shape[-1], bias=False, initializer="linear",
                    name="atom37_encoder_time_linear")(time)
            local37 = hk.LayerNorm([-1], True, True, name="atom37_encoder_input_norm")(local37)
            if getattr(self.config, "trace", False):
                return (local, pos.to_array(),
                        local37, pos37.to_array(),
                        resi, chain, batch, mask,
                        neighbours, local_features,
                        neighbours37, local_features37)
            return local, pos.to_array(), local37, pos37.to_array(), resi, chain, batch, mask
        if getattr(self.config, "trace", False):
            return local, pos.to_array(), resi, chain, batch, mask, neighbours, local_features
        return local, pos.to_array(), resi, chain, batch, mask

class Decoder(hk.Module):
    """Protein structure decoder."""
    def __init__(self, config, name: Optional[str] = "diffusion"):
        super().__init__(name)
        self.config = config
        self.decoder_block = DecoderBlock

    def __call__(self, data, prev):
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        sidechain_decoder = getattr(c, "sidechain_decoder", "angles")
        # select a decoder block (equivariant or non-equivariant)
        decoder_module = DecoderStack
        if c.equivariance == "nonequivariant":
            decoder_module = NonEquivariantDecoderStack
            self.decoder_block = NonEquivariantDecoderBlock
        elif c.equivariance == "semiequivariant":
            decoder_module = SemiEquivariantDecoderStack
            self.decoder_block = SemiEquivariantDecoderBlock
        if atom37_parallel_mode == "parallel" and c.equivariance in ("nonequivariant", "semiequivariant"):
            raise ValueError(
                "atom37_parallel_mode='parallel' currently supports only the "
                "equivariant decoder path.")
        decoder_stack = decoder_module(c, self.decoder_block)
        decoder_stack37 = None
        if atom37_parallel_mode == "parallel":
            decoder_stack37 = DecoderStackAtom37(c, DecoderBlockAtom37)
        aa_decoder = AADecoder(c)
        # run the decoder
        prep = self.prepare_features(data, prev)
        if atom37_parallel_mode == "parallel":
            local, pos, local37, pos37, resi, chain, batch, mask = prep
        else:
            local, pos, resi, chain, batch, mask = prep
        sup_neighbours = get_random_neighbours(c.fape_neighbours)(
            Vec3Array.from_array(data["pos_gt"][:, -1]), batch, mask)
        local, pos, trajectory, sup_distogram = decoder_stack(
            local, pos, resi, chain, batch, mask,
            sup_neighbours)
        trajectory37 = None
        if atom37_parallel_mode == "parallel":
            local37, pos37, trajectory37 = decoder_stack37(
                local37, pos37, resi, chain, batch, mask)
        # predict & handle sequence, losses etc.
        result = dict()
        result["latent"] = data["latent"]
        result["trajectory"] = trajectory
        result["sup_neighbours"] = sup_neighbours
        result["sup_distogram"] = sup_distogram
        result["local"] = local
        result["pos"] = pos
        if atom37_parallel_mode == "parallel":
            result["trajectory37"] = trajectory37
            result["local37"] = local37
            result["pos37"] = pos37
        if c.input_diffusion or c.latent_diffusion:
            latent_decoder = MLP(
                c.local_size * 2,
                c.local_size if c.noembed else c.latent_size,
                bias=False,
                final_init=init_zeros(), activation=jax.nn.gelu,
                name="latent_decoder")
            latent_update = latent_decoder(
                jnp.concatenate((
                    #local,
                    hk.LayerNorm([-1], True, True)(local),
                    data["latent"]), axis=-1))
            time = data["time"]
            if c.vp_diffusion:
                predicted_latent = data["latent"] + latent_update
            else:
                predicted_latent = data["skip_latent"] + latent_update * time[:, None] / jnp.sqrt(1 + time[:, None] ** 2)
            result["predicted_latent"] = predicted_latent
        # decoder features and logits
        aa_logits, decoder_features, corrupt_aa = aa_decoder.train(
            data["aa_gt"], local, pos, resi, chain, batch, mask)
        result["aa"] = aa_logits
        result["aa_features"] = decoder_features
        result["corrupt_aa"] = corrupt_aa * data["mask"] * (data["aa_gt"] != 20)
        result["aa_gt"] = data["aa_gt"]
        if sidechain_decoder == "angles":
            # generate all-atom positions using predicted
            # side-chain torsion angles:
            aatype = data["aa_gt"]
            if c.eval:
                aatype = jnp.argmax(aa_logits, axis=-1)
            raw_angles, angles, angle_pos = get_angle_positions(
                aatype, local, pos)
            result["raw_angles"] = raw_angles
            result["angles"] = angles
            result["atom_pos"] = angle_pos
        elif sidechain_decoder == "atom14_masked":
            atom_head_input = getattr(c, "atom14_masked_input", "local")
            atom_head_features = local
            if atom_head_input in ("concat", "fusion"):
                atom_head_features = jnp.concatenate(
                    (local, decoder_features), axis=-1)
            elif atom_head_input != "local":
                raise ValueError(
                    f"Unknown atom14 masked input mode: {atom_head_input}")
            atom_backbone_source = getattr(
                c, "atom14_masked_backbone_source", "predicted")
            atom_head_pos = pos
            if atom_backbone_source == "gt" and not c.eval:
                atom_head_pos = jnp.concatenate((
                    data["pos_gt"][..., :4, :],
                    pos[..., 4:5, :]
                ), axis=-2)
            elif atom_backbone_source not in ("predicted", "gt"):
                raise ValueError(
                    f"Unknown atom14 masked backbone source: {atom_backbone_source}")
            atom_pos_raw, sidechain_local_raw = Atom14MaskedHead(c)(atom_head_features, atom_head_pos)
            aatype_for_mask = data["aa_gt"]
            if c.eval:
                aatype_for_mask = jnp.argmax(aa_logits, axis=-1)
            atom_mask_pred = get_atom14_mask(aatype_for_mask) * data["mask"][:, None]
            atom_pos = atom_pos_raw * atom_mask_pred[..., None]
            result["atom_head_pos"] = atom_head_pos
            result["aatype_for_mask"] = aatype_for_mask
            result["atom_mask_pred"] = atom_mask_pred
            result["sidechain_local_raw"] = sidechain_local_raw
            result["atom_pos_raw"] = atom_pos_raw
            result["atom_pos"] = atom_pos
        else:
            raise ValueError(f"Unknown sidechain decoder mode: {sidechain_decoder}")
        return result

    def init_prev(self, data):
        """Initialize recycling inputs."""
        c = self.config
        prev = dict(
            pos = data["pos"],
            local = jnp.zeros((data["pos"].shape[0], c.local_size), dtype=jnp.float32)
        )
        if getattr(c, "atom37_parallel_mode", "none") == "parallel":
            prev.update(
                pos37=data["pos37"],
                local37=jnp.zeros((data["pos37"].shape[0], c.local_size), dtype=jnp.float32))
        return prev

    def prepare_features(self, data, prev):
        """Construct Decoder input features from a batch of data and recycling information."""
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        # directly use the recycled positions as a starting point
        pos = prev["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        latent = data["latent"]
        mask = data["mask"]
        _, local_pos = extract_aa_frames(
            Vec3Array.from_array(pos))
        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            latent
        ]
        # add time embedding if trained as a diffusoin model (not used in manuscript)
        if c.time_embedding and c.latent_diffusion and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 80.0, bins=200)
            local_features.append(time)
        local_features.append(hk.LayerNorm([-1], True, True)(prev["local"]))
        local_features = jnp.concatenate(
            local_features, axis=-1)
        local = MLP(
            4 * c.local_size, c.local_size,
            activation=jax.nn.gelu,
            bias=False,
            final_init=init_linear())(local_features)
        local = hk.LayerNorm([-1], True, True)(local)
        if atom37_parallel_mode == "parallel":
            pos37 = prev["pos37"]
            _, local_pos37 = extract_aa_frames_atom37(
                Vec3Array.from_array(pos37))
            local_features37 = [
                local_pos37.to_array().reshape(local_pos37.shape[0], -1),
                distance_rbf(local_pos37.norm(), 0.0, 22.0).reshape(local_pos37.shape[0], -1),
                latent
            ]
            if c.time_embedding and c.latent_diffusion and "time" in data:
                time = data["time"]
                time = distance_rbf(time, 0, 80.0, bins=200)
                local_features37.append(time)
            local_features37.append(
                hk.LayerNorm([-1], True, True, name="atom37_prev_local_norm")(prev["local37"]))
            local_features37 = jnp.concatenate(local_features37, axis=-1)
            local37 = MLP(
                4 * c.local_size, c.local_size,
                activation=jax.nn.gelu,
                bias=False,
                final_init=init_linear(),
                name="atom37_decoder_input")(local_features37)
            local37 = hk.LayerNorm([-1], True, True, name="atom37_decoder_input_norm")(local37)
            return local, pos, local37, pos37, resi, chain, batch, mask
        return local, pos, resi, chain, batch, mask

    def loss(self, data, result):
        """Compute Decoder losses from the input data and Decoder results."""
        c = self.config
        atom37_parallel_mode = getattr(c, "atom37_parallel_mode", "none")
        sidechain_decoder = getattr(c, "sidechain_decoder", "angles")
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]
        losses = dict()
        total = 0.0
        
        # AA NLL loss
        aa_predict_mask = mask * (data["aa_gt"] != 20)
        losses["debug_mask_sum"] = mask.sum()
        losses["debug_aa_mask_sum"] = aa_predict_mask.sum()
        aa_nll = -(result["aa"] * jax.nn.one_hot(data["aa_gt"], 20, axis=-1)).sum(axis=-1)
        aa_nll = jnp.where(aa_predict_mask, aa_nll, 0)
        aa_nll = aa_nll.sum() / jnp.maximum(aa_predict_mask.sum(), 1)
        losses["aa"] = aa_nll
        losses["weighted_aa"] = c.aa_weight * aa_nll
        total += losses["weighted_aa"]

        # position losses
        base_weight = mask / jnp.maximum(index_sum(mask.astype(jnp.float32), batch, mask), 1) / (batch.max() + 1)
        losses["debug_base_weight_sum"] = base_weight.sum()

        # sparse neighbour FAPE ** 2
        pair_mask = batch[:, None] == batch[None, :]
        pair_mask *= mask[:, None] * mask[None, :]
        pos_gt = data["pos_gt"]
        pos_gt = jnp.where(mask[:, None, None], pos_gt, 0)
        pos_gt = Vec3Array.from_array(pos_gt)
        frames_gt, _ = extract_aa_frames(jax.lax.stop_gradient(pos_gt))
        # CB distance
        distance = data["dmap"]
        distance = jnp.where(pair_mask, distance, jnp.inf)
        # get random neighbours to compute sparse FAPE on
        neighbours = get_random_neighbours(c.fape_neighbours)(distance, batch, mask)
        losses["debug_neighbours_hash"] = jnp.sum(neighbours)
        mask_neighbours = (neighbours != -1) * mask[:, None] * mask[neighbours]
        pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt[neighbours])
        traj = Vec3Array.from_array(result["trajectory"])
        frames, _ = jax.vmap(extract_aa_frames)(traj)
        traj_local = frames[:, :, None, None].apply_inverse_to_point(traj[:, neighbours])
        fape_base = (traj_local - pos_gt_local).norm2()
        fape_clipped = jnp.clip(fape_base, 0.0, c.clip_fape)
        if c.unclipped_weight:
            fape_clipped = fape_base * c.unclipped_weight + fape_clipped
        if c.no_fape2:
            # use FAPE instead of FAPE ** 2
            fape_clipped = jnp.sqrt(jnp.maximum(fape_clipped, 1e-6))
        fape_traj = fape_clipped.mean(axis=-1)
        fape_traj = jnp.where(mask_neighbours[None], fape_traj, 0)
        fape_traj = fape_traj.sum(axis=-1) / jnp.maximum(mask_neighbours.sum(axis=1)[None], 1)
        fape_traj = (fape_traj * base_weight).sum(axis=-1)
        losses["fape"] = fape_traj[-1] / 3
        losses["fape_trajectory"] = fape_traj.mean() / 3
        fape_loss = (c.fape_weight * fape_traj[-1] + c.fape_trajectory_weight * fape_traj.mean()) / 3
        losses["weighted_fape"] = fape_loss
        total += losses["weighted_fape"]

        if atom37_parallel_mode == "parallel":
            pos37 = Vec3Array.from_array(result["pos37"])
            pos37_gt = Vec3Array.from_array(data["pos37_gt"])
            local_neighbours37 = get_spatial_neighbours(
                count=c.local_neighbours)(
                    pos37_gt[:, 1], batch, mask)
            mask37_gt = data["all_atom_mask_37"][local_neighbours37]
            frames37_gt, _ = extract_aa_frames_atom37(pos37_gt)
            pos37_gt_local = frames37_gt[:, None, None].apply_inverse_to_point(
                pos37_gt[local_neighbours37])
            frames37, _ = extract_aa_frames_atom37(pos37)
            pos37_local = frames37[:, None, None].apply_inverse_to_point(
                pos37[local_neighbours37])
            pos37_loss = jnp.where(
                mask37_gt, (pos37_local - pos37_gt_local).norm2(), 0).sum(axis=(1, 2))
            pos37_loss /= jnp.maximum(mask37_gt.sum(axis=(1, 2)), 1)
            pos37_loss = (jnp.where(mask, pos37_loss, 0) * base_weight).sum() / 3
            losses["pos37"] = pos37_loss
            losses["pos37_local"] = pos37_loss
            weighted_pos37 = getattr(c, "pos37_local_weight", 0.0) * pos37_loss
            losses["weighted_pos37_local"] = weighted_pos37
            total += weighted_pos37

            # Diagnostic RMSD for atom37 branch after rigid (Kabsch) alignment.
            # These metrics are logging-only and never added to `total`.
            pos37_gt_aligned = index_align(
                pos37_gt, pos37, batch, mask)
            atom37_mask = data["all_atom_mask_37"] * mask[:, None]
            atom37_sqerr = (pos37.to_array() - pos37_gt_aligned.to_array()) ** 2
            atom37_dist2 = atom37_sqerr.sum(axis=-1)
            atom37_count = jnp.maximum(atom37_mask.sum(), 1)
            losses["diag_rmsd_atom37"] = jnp.sqrt(
                jnp.where(atom37_mask, atom37_dist2, 0).sum() / atom37_count)

            bb_mask = jnp.zeros_like(atom37_mask, dtype=jnp.bool_)
            bb_mask = bb_mask.at[:, 0].set(True)  # N
            bb_mask = bb_mask.at[:, 1].set(True)  # CA
            bb_mask = bb_mask.at[:, 2].set(True)  # C
            bb_mask = bb_mask.at[:, 4].set(True)  # O
            bb_mask = bb_mask * atom37_mask
            bb_count = jnp.maximum(bb_mask.sum(), 1)
            losses["diag_rmsd_bb_atom37"] = jnp.sqrt(
                jnp.where(bb_mask, atom37_dist2, 0).sum() / bb_count)

            sc_mask = (~bb_mask) * atom37_mask
            sc_count = jnp.maximum(sc_mask.sum(), 1)
            losses["diag_rmsd_sc_atom37"] = jnp.sqrt(
                jnp.where(sc_mask, atom37_dist2, 0).sum() / sc_count)

        # sup distogram loss
        if c.distogram_block != "none":
            cb_gt = pos_gt[:, -1]
            sup_neighbours = result["sup_neighbours"]
            sup_mask = (sup_neighbours != -1) * mask[:, None] * mask[sup_neighbours]
            losses["debug_sup_mask_sum"] = sup_mask.sum()
            dist_gt = (cb_gt[:, None] - cb_gt[sup_neighbours]).norm()
            dist_one_hot = distance_one_hot(dist_gt, 0, 22.0, 16)
            distogram_nll = -(result["sup_distogram"] * dist_one_hot[None]).sum(axis=-1)
            distogram_nll = jnp.where(sup_mask, distogram_nll, 0).sum(axis=-1)
            distogram_nll /= jnp.maximum(sup_mask.sum(axis=1), 1)
            distogram_nll = (distogram_nll * base_weight).sum(axis=1)
            # check what is in the last index
            losses["distogram"] = distogram_nll[-1]
            losses["distogram_trajectory"] = distogram_nll.mean()
            losses["weighted_distogram"] = 10.0 * distogram_nll[-1] + 5.0 * distogram_nll.mean()
            total += losses["weighted_distogram"]

        # Kabsch RMSD loss
        if c.kabsch_rmsd:
            last = traj[-1]
            pos_gt = jax.lax.stop_gradient(index_align(pos_gt, last, batch, mask))
            pos_loss_unclipped = (pos_gt[None] - traj).norm2()
            pos_loss_clipped = jnp.clip(pos_loss_unclipped, 0, 100.0)
            pos_loss = pos_loss_clipped
            if c.unclipped_weight:
                pos_loss = pos_loss_unclipped * c.unclipped_weight + pos_loss
            pos_loss = pos_loss.mean(axis=-1)
            pos_loss *= base_weight[None]
            pos_loss = pos_loss.sum(axis=-1)
            losses["kabsch_rmsd"] = pos_loss[-1] / 3
            losses["kabsch_rmsd_trajectory"] = pos_loss.mean() / 3
            pos_loss = (c.fape_weight * pos_loss[-1] + c.fape_trajectory_weight * pos_loss.mean()) / 3
            losses["weighted_kabsch_rmsd"] = pos_loss
            total += losses["weighted_kabsch_rmsd"]

        if sidechain_decoder in ("angles", "atom14_masked"):
            # local loss. For the masked atom14 decoder we supervise raw coordinates
            # with the GT atom mask so the geometric regularizer stays independent
            # of the post-hoc sequence-based masking step.
            atom_pos_source = result["atom_pos"]
            if sidechain_decoder == "atom14_masked":
                atom_pos_source = result["atom_pos_raw"]
            atom_pos = Vec3Array.from_array(atom_pos_source)
            atom_pos_gt = Vec3Array.from_array(data["atom_pos"])
            local_neighbours = get_spatial_neighbours(
                count=c.local_neighbours)(
                    pos_gt[:, -1], batch, mask)
            mask_gt = data["atom_mask"][local_neighbours]
            frames_gt, _ = extract_aa_frames(atom_pos_gt)
            atom_pos_gt = frames_gt[:, None, None].apply_inverse_to_point(atom_pos_gt[local_neighbours])
            frames, _ = extract_aa_frames(atom_pos)
            atom_pos = frames[:, None, None].apply_inverse_to_point(atom_pos[local_neighbours])
            local_loss = jnp.where(mask_gt, (atom_pos - atom_pos_gt).norm2(), 0).sum(axis=(1, 2))
            local_loss /= jnp.maximum(mask_gt.sum(axis=(1, 2)), 1)
            local_loss = (jnp.where(mask, local_loss, 0) * base_weight).sum() / 3
            losses["local"] = local_loss
            losses["weighted_local"] = c.local_weight * local_loss
            total += losses["weighted_local"]

        if sidechain_decoder == "atom14_masked":
            full_atom_loss_mode = getattr(c, "full_atom_loss_mode", "local")
            if full_atom_loss_mode == "local":
                _, sidechain_local_gt = extract_aa_frames(
                    Vec3Array.from_array(data["atom_pos"]))
                sidechain_local_gt = sidechain_local_gt.to_array()[..., 4:, :]
                sidechain_sqerr = (result["sidechain_local_raw"] - sidechain_local_gt) ** 2
                sidechain_mask = data["atom_mask"][..., 4:]
                sidechain_sqerr = jnp.where(sidechain_mask[..., None], sidechain_sqerr, 0)
                sidechain_sqerr = sidechain_sqerr.sum(axis=(-1, -2))
                sidechain_count = jnp.maximum(sidechain_mask.sum(axis=-1), 1)
                full_atom_l2 = sidechain_sqerr / sidechain_count
            elif full_atom_loss_mode == "global":
                sidechain_sqerr = (result["atom_pos_raw"][..., 4:, :] - data["atom_pos"][..., 4:, :]) ** 2
                sidechain_mask = data["atom_mask"][..., 4:]
                sidechain_sqerr = jnp.where(sidechain_mask[..., None], sidechain_sqerr, 0)
                sidechain_sqerr = sidechain_sqerr.sum(axis=(-1, -2))
                sidechain_count = jnp.maximum(sidechain_mask.sum(axis=-1), 1)
                full_atom_l2 = sidechain_sqerr / sidechain_count
            else:
                raise ValueError(
                    f"Unknown full atom loss mode: {full_atom_loss_mode}")
            full_atom_l2 = (jnp.where(mask, full_atom_l2, 0) * base_weight).sum() / 3
            losses["full_atom_l2"] = full_atom_l2
            losses["weighted_full_atom_l2"] = getattr(c, "full_atom_weight", 0.0) * full_atom_l2
            total += losses["weighted_full_atom_l2"]

        # VQ losses
        if c.codebook_size and not c.is_decoder:
            cl = result["codebook_losses"]
            losses.update(cl)
            if not c.state:
                total += c.codebook_loss_scale * (cl["codebook"] + cl["unassigned"])
            total += c.codebook_loss_scale * c.codebook_b * cl["commitment"]

        # additional denoising losses (not used in manuscript)
        if (c.input_diffusion or c.latent_diffusion) and c.latent_loss_scale:
            raw_loss = ((data["clean_latent"] - result["predicted_latent"]) ** 2).mean(axis=-1)
            if c.vp_diffusion:
                weighted_loss = (jnp.where(mask, raw_loss, 0) * base_weight).sum()
            else:
                time = jnp.maximum(data["time"], 1e-2)
                raw_loss = raw_loss * (1 + time ** 2) / jnp.maximum(time ** 2, 1e-6)
                weighted_loss = (jnp.where(mask, raw_loss, 0) * base_weight).sum()
            losses["latent"] = weighted_loss
            losses["weighted_latent"] = c.latent_loss_scale * weighted_loss
            total += losses["weighted_latent"]
        # AlphaFold violation loss. (not used in manuscript)
        if c.violation_scale and sidechain_decoder == "angles":
            res_mask = data["mask"]
            pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
            violation, _ = violation_loss(data["aa_gt"],
                                          data["residue_index"],
                                          result["atom_pos"],
                                          pred_mask,
                                          res_mask,
                                          clash_overlap_tolerance=1.5,
                                          violation_tolerance_factor=2.0,
                                          chain_index=data["chain_index"],
                                          batch_index=data["batch_index"],
                                          per_residue=False)
            losses["violation"] = violation.mean()
            losses["weighted_violation"] = c.violation_scale * violation.mean()
            total += losses["weighted_violation"]

        losses["total"] = total
        return total, losses

def get_angle_positions(aa_gt, local, pos):
    """Construct side chain atom positions from amino acid sequence and features."""
    frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
    features = [
        local,
        local_positions.to_array().reshape(local_positions.shape[0], -1),
        distance_rbf(local_positions.norm(),
                     0.0, 10.0, 16).reshape(local_positions.shape[0], -1),
        jax.nn.one_hot(aa_gt, 21, axis=-1)
    ]
    raw_angles = MLP(
        local.shape[-1] * 2, 7 * 2, bias=False,
        activation=jax.nn.gelu, final_init="linear")(
            jnp.concatenate(features, axis=-1))

    raw_angles = raw_angles.reshape(-1, 7, 2)
    angles = raw_angles / jnp.sqrt(jnp.maximum(
        (raw_angles ** 2).sum(axis=-1, keepdims=True), 1e-6))
    angle_pos, _ = single_protein_sidechains(
        aa_gt, frames, angles)
    angle_pos = angle_pos.to_array().reshape(-1, 14, 3)
    angle_pos = jnp.concatenate((
        pos[..., :4, :],
        angle_pos[..., 4:, :]
    ), axis=-2)
    return raw_angles, angles, angle_pos

class Atom14MaskedHead(hk.Module):
    """Predict raw atom14 coordinates without conditioning on amino acid identity."""
    def __init__(self, config, name: Optional[str] = "atom14_masked_head"):
        super().__init__(name)
        self.config = config

    def __call__(self, atom_features, pos):
        c = self.config
        atom_head_input = getattr(c, "atom14_masked_input", "local")
        if atom_head_input == "fusion":
            atom_features = MLP(
                c.local_size * 2, c.local_size, bias=False,
                activation=jax.nn.gelu, final_init=init_linear(),
                name="atom14_feature_fusion")(atom_features)
            atom_features = hk.LayerNorm([-1], True, True)(atom_features)
        frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
        features = [
            atom_features,
            local_positions.to_array().reshape(local_positions.shape[0], -1),
            distance_rbf(local_positions.norm(),
                         0.0, 10.0, 16).reshape(local_positions.shape[0], -1),
        ]
        raw_sidechain_local = MLP(
            atom_features.shape[-1] * 2, 10 * 3, bias=False,
            activation=jax.nn.gelu, final_init="linear")(
                jnp.concatenate(features, axis=-1))
        raw_sidechain_local = raw_sidechain_local.reshape(-1, 10, 3)
        raw_sidechain_global = frames[..., None].apply_to_point(
                Vec3Array.from_array(raw_sidechain_local)).to_array()
        raw_atom_pos = jnp.concatenate((
            pos[..., :4, :],
            raw_sidechain_global
        ), axis=-2)
        return raw_atom_pos, raw_sidechain_local

class AADecoder(hk.Module):
    """Amino acid sequence decoder module."""
    def __init__(self, config, name: Optional[str] = "aa_decoder"):
        super().__init__(name)
        self.config = config

    def __call__(self, aa, local, pos, resi, chain, batch, mask):
        c = self.config
        neighbours = get_spatial_neighbours(32)(
            Vec3Array.from_array(pos)[:, -1], batch, mask)
        local = AADecoderStack(c, depth=c.aa_decoder_depth)(
            aa, local, pos, neighbours,
            resi, chain, batch, mask)
        local = hk.LayerNorm([-1], True, True)(local)
        return Linear(20, initializer=init_zeros(), bias=False)(local), local

    def train(self, aa, local, pos, resi, chain, batch, mask):
        """Apply the AADecoder during training."""
        c = self.config
        aa = 20 * jnp.ones_like(aa)
        logits, features = self(
            aa, local, pos, resi, chain, batch, mask)
        logits = jax.nn.log_softmax(logits)
        return logits, features, jnp.ones_like(mask)

class DecoderStack(hk.Module):
    """Standard stack of Decoder blocks."""
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos,
                 resi, chain, batch, mask,
                 sup_neighbours):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the decoder block
                features, pos, sup_distogram = block(c)(
                    features, pos,
                    resi, chain, batch, mask,
                    sup_neighbours)
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), (trajectory_output, sup_distogram)
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        (local, pos), (trajectory, sup_distogram) = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
            sup_distogram = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], sup_distogram))
        return local, pos, trajectory, sup_distogram


class DecoderStackAtom37(hk.Module):
    """Parallel atom37 decoder stack with trajectory output."""
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack_atom37"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local37, pos37,
                 resi, chain, batch, mask):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                features, pos = block(c)(
                    features, pos,
                    resi, chain, batch, mask)
                trajectory_output = pos
                return (features, pos), trajectory_output
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        (local37, pos37), trajectory37 = stack((local37, pos37))
        if c.block_size > 1:
            trajectory37 = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), trajectory37)
        return local37, pos37, trajectory37


class SemiEquivariantDecoderStack(hk.Module):
    """Stack of partly nonequivariant Decoder blocks using data augmentation."""
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos,
                 resi, chain, batch, mask,
                 sup_neighbours):
        c = self.config
        def stack_inner(block):
            def _inner(data):
                features, pos = data
                # run the decoder block
                features, pos, sup_distogram = block(c)(
                    features, pos,
                    resi, chain, batch, mask,
                    sup_neighbours)
                trajectory_output = pos
                # return features & positions for the next block
                # and positions to construct a trajectory
                return (features, pos), (trajectory_output, sup_distogram)
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        # apply data augmentation when training non-equivariant Decoders
        pos = structure_augmentation(pos / c.sigma_data, batch, mask) * c.sigma_data
        (local, pos), (trajectory, sup_distogram) = stack((local, pos))
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
            sup_distogram = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], sup_distogram))
        return local, pos, trajectory, sup_distogram

class NonEquivariantDecoderStack(hk.Module):
    """Stack of fully non-equivariant decoder blocks."""
    def __init__(self, config, block,
                 name: Optional[str] = "decoder_stack"):
        super().__init__(name)
        self.config = config
        self.block = block

    def __call__(self, local, pos,
                 resi, chain, batch, mask,
                 sup_neighbours):
        c = self.config
        def stack_inner(block):
            def _inner(localpos):
                # run the decoder block
                localpos, sup_distogram = block(c)(
                    localpos,
                    resi, chain, batch, mask,
                    sup_neighbours)
                # return features & positions for the next block
                # and positions to construct a trajectory
                return localpos, (localpos, sup_distogram)
            return _inner
        decoder_block = self.block
        stack = block_stack(
            c.depth, c.block_size, with_state=True)(
                hk.remat(stack_inner(decoder_block)))
        augment_pos = structure_augmentation(pos / c.sigma_data, batch, mask)
        localpos = Linear(c.local_size, bias=False, initializer="linear")(
            jnp.concatenate((local, augment_pos.reshape(pos.shape[0], -1)), axis=-1)
        )
        localpos, (trajectory, sup_distogram) = stack(localpos)
        if c.block_size > 1:
            # if the block-size for nested gradient checkpointing
            # is > 1, unwrap the nested trajectory from shape
            # (depth / block_size, block_size, ...) to shape
            # (depth, ...)
            trajectory = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], trajectory))
            sup_distogram = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:], sup_distogram))
        pos_project = Linear(5 * 3, bias=False)
        trajectory = hk.LayerNorm([-1], True, True)(trajectory)
        trajectory = pos_project(trajectory).reshape(
            trajectory.shape[0], trajectory.shape[1], 5, 3)
        trajectory *= c.sigma_data
        pos = trajectory[-1]
        return local, pos, trajectory, sup_distogram

def structure_augmentation_params(pos, batch, mask):
    """Sample random rotation and translation parameters for data augmentation."""
    # center positions
    center = index_mean(pos[:, 1], batch, mask[:, None])
    # centering + random translation
    translation = jax.random.normal(hk.next_rng_key(), pos[:, 1].shape)[batch]
    # random rotation
    rotation = random_rotation(batch)
    return center, translation, rotation

def apply_structure_augmentation(pos, center, translation, rotation):
    """Apply a rotation and translation to an array of atom positions."""
    # center positions
    pos -= center[:, None]
    # apply transformation
    pos = rotation[:, None].apply_to_point(Vec3Array.from_array(pos)).to_array()
    pos += translation[:, None]
    return pos

def apply_inverse_structure_augmentation(pos, center, translation, rotation):
    """Apply the inverse of a rotation and translation to an array of atom positions."""
    # invert translation
    pos -= translation[:, None]
    # invert rotation
    pos = rotation[:, None].apply_inverse_to_point(Vec3Array.from_array(pos)).to_array()
    # move to center
    pos += center[:, None]
    return pos

def structure_augmentation(pos, batch, mask):
    """Randomly rotate and translate an array of atom positions."""
    # get random augmentation parameters
    center, translation, rotation = structure_augmentation_params(pos, batch, mask)
    # apply random augmentation
    pos = apply_structure_augmentation(pos, center, translation, rotation)
    return pos

def random_rotation(batch):
    """Sample a random rotation."""
    x = jax.random.normal(hk.next_rng_key(), (batch.shape[0], 3))[batch]
    y = jax.random.normal(hk.next_rng_key(), (batch.shape[0], 3))[batch]
    result = Rot3Array.from_two_vectors(
        Vec3Array.from_array(x),
        Vec3Array.from_array(y))
    return result

def aa_decoder_pair_features(c):
    """Pair features for the AADecoder module."""
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        return pair, pair_mask
    return inner


def aa_decoder_pair_features_atom37(c):
    """Pair features for atom37 branch in the AADecoder/Encoder modules."""
    def inner(pos37, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pos_view = Vec3Array.from_array(atom37_to_ncacocb(pos37.to_array()))
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos_view, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos_view, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos_view, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos_view, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        return pair, pair_mask
    return inner


def init_local_features(c):
    """Initial structure embedding for the Encoder."""
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(8, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.gelu)(pair)
        pair = jnp.where(pair_mask[..., None], pair, 0)
        local = pair.sum(axis=1) / jnp.maximum(pair_mask.sum(axis=-1)[..., None], 1)
        local = Linear(c.local_size, bias=False)(local)
        return local
    return inner


def init_local_features_atom37(c):
    """Initial structure embedding for the parallel atom37 encoder."""
    def inner(pos37, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pos_view = Vec3Array.from_array(atom37_to_ncacocb(pos37.to_array()))
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(8, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos_view, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos_view, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos_view, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos_view, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(c.pair_size * 2, c.pair_size, activation=jax.nn.gelu)(pair)
        pair = jnp.where(pair_mask[..., None], pair, 0)
        local = pair.sum(axis=1) / jnp.maximum(pair_mask.sum(axis=-1)[..., None], 1)
        local = Linear(c.local_size, bias=False)(local)
        return local
    return inner


def decoder_pair_features(c):
    """Pair features for the equivariant Decoder."""
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                        distance_rbf(dmap), 0))
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner


def decoder_pair_features_atom37(c):
    """Pair features for the parallel atom37 decoder branch."""
    def inner(pos37, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                          distance_rbf(dmap), 0))
        pos_view = Vec3Array.from_array(atom37_to_ncacocb(pos37))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos_view, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos_view, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos_view, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos_view, neighbours))
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner


def hybrid_decoder_pair_features(c, center_features=False):
    """Pair features for the partially nonequivariant Decoder. (NEQ in the manuscript)"""
    def inner(pos, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                        distance_rbf(dmap), 0))
        pos = Vec3Array.from_array(pos)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        local_neighbourhood = jnp.concatenate((
            jnp.repeat(pos.to_array()[:, None], neighbours.shape[1], axis=1),
            pos.to_array()[neighbours]
        ), axis=-2) / c.sigma_data
        center_neighbourhood = local_neighbourhood -  pos.to_array()[:, None, 1, None]
        center_neighbourhood = (Vec3Array.from_array(center_neighbourhood).normalized()).to_array()
        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        center_neighbourhood = center_neighbourhood.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            jnp.concatenate((local_neighbourhood, center_neighbourhood), axis=-1))
        dirs = (pos[:, None, :, None] - pos[neighbours, None, :]).to_array()
        dirs = dirs.reshape(*neighbours.shape, -1)
        pair += Linear(c.pair_size, bias=False, initializer="linear")(dirs)

        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1], activation=jax.nn.gelu, final_init="linear")(pair)
        return pair, pair_mask
    return inner

def nonequivariant_decoder_pair_features(c):
    """Pair features for a fully non-equivariant decoder.
    
    Not used in the manuscript.
    """
    def inner(local, dmap, neighbours,
              resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        index = axis_index(neighbours, axis=0)
        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True)(
                resi, chain, batch, neighbours))
        if dmap is not None:
            pair += Linear(c.pair_size, initializer="linear", bias=False)(
                jnp.where(pair_mask[..., None],
                        distance_rbf(dmap), 0))
        pair += Linear(c.pair_size, bias=False)(local)[:, None]
        pair += Linear(c.pair_size, bias=False)(local)[neighbours]
        pair = hk.LayerNorm([-1], True, True)(pair)
        pair = MLP(pair.shape[-1] * 2, pair.shape[-1],
                   activation=jax.nn.gelu,
                   final_init="linear")(pair)
        return pair, pair_mask
    return inner

def extract_dmap_neighbours(count=32):
    """Extract nearest neighbours from a distance map."""
    def inner(distance, resi, chain, batch, mask):
        same_item = batch[:, None] == batch[None, :]
        weight = -3 * jnp.log(distance + 1e-6)
        uniform = jax.random.uniform(
            hk.next_rng_key(), weight.shape,
            dtype=weight.dtype, minval=1e-6, maxval=1 - 1e-6)
        gumbel = jnp.log(-jnp.log(uniform))
        weight = weight - gumbel
        distance = -weight
        distance = jnp.where(same_item, distance, jnp.inf)
        mask = (mask[:, None] * mask[None, :] * same_item)
        return get_neighbours(count)(distance, mask)
    return inner

def extract_neighbours(num_index=16, num_spatial=16, num_random=16):
    """Extract nearest neighbours of a residue based on sequence and euclidean distance."""
    def inner(pos, resi, chain, item, mask):
        pos = pos[:, 1]
        same_batch = (item[:, None] == item[None, :])
        same_chain = (chain[:, None] == chain[None, :])
        valid = same_batch * (mask[:, None] * mask[None, :])
        within = abs(resi[:, None] - resi[None, :]) < num_index
        within *= same_batch * same_chain
        distance = (pos[:, None] - pos[None, :]).norm()
        distance = jnp.where(within, jnp.inf, distance)
        distance = jnp.where(valid, distance, jnp.inf)
        cutoff = jnp.sort(distance)[:, :num_spatial][:, -1]
        within = within + (distance < cutoff) > 0
        random_distance = -3 * jnp.log(jnp.maximum(distance, 1e-6))
        gumbel = jax.random.gumbel(hk.next_rng_key(), random_distance.shape)
        random_distance = jnp.where(within, -10_000, -(random_distance - gumbel))
        random_distance = jnp.where(valid, random_distance, jnp.inf)
        neighbours = get_neighbours(
            num_index + num_spatial + num_random)(
                random_distance, mask)
        return neighbours
    return inner

class VQ(hk.Module):
    """Vector quantization module."""
    def __init__(self, codebook_size=4096,
                 affine=None,
                 mapped_axes=None,
                 name: str | None = "vq"):
        super().__init__(name)
        self.codebook_size = codebook_size
        self.affine = affine
        self.mapped_axes = mapped_axes if mapped_axes else []

    def __call__(self, features, mask):
        def replace_gradient(x, y):
            return jax.lax.stop_gradient(x) + y - jax.lax.stop_gradient(y)
        codebook = 10.0 * hk.get_parameter("codebook",
                                    (self.codebook_size, features.shape[-1]),
                                    init=hk.initializers.RandomUniform(-0.1, 0.1))
        if self.affine:
            codebook_mean = hk.get_parameter("codebook_mean",
                                             (1, features.shape[-1]),
                                             init=hk.initializers.Constant(0.0))
            codebook_scale = hk.get_parameter("codebook_scale",
                                              (1, features.shape[-1]),
                                              init=hk.initializers.Constant(0.0))
            codebook_scale = jnp.exp(codebook_scale)
            codebook = codebook_mean + codebook_scale * codebook
        distance = ((features[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
        distance = jnp.where(mask[:, None], distance, jnp.inf)
        assign_fwd = jnp.argmin(distance, axis=1)
        assign_rev = jnp.argmin(distance, axis=0)
        features_fwd = codebook[assign_fwd]
        features_rev = features[assign_rev]
        out_features = replace_gradient(features_fwd, features)
        codebook_loss = ((features_fwd - jax.lax.stop_gradient(features)) ** 2).mean(axis=-1)
        codebook_loss = jnp.where(mask, codebook_loss, 0).sum() / jnp.maximum(mask.sum(), 1)
        commitment_loss = ((jax.lax.stop_gradient(features_fwd) - features) ** 2).mean(axis=-1)
        commitment_loss = jnp.where(mask, commitment_loss, 0).sum() / jnp.maximum(mask.sum(), 1)
        assignment_count = jnp.zeros((self.codebook_size,),
                                     dtype=jnp.float32).at[assign_fwd].add(mask)
        # if we are mapping over one or more axes
        # sum assignment_count over all of them
        local_assignment_count = assignment_count
        if (not hk.running_init()):
            for axis in self.mapped_axes:
                assignment_count = jax.lax.psum(assignment_count, axis)
        assignment_mask = local_assignment_count < 1
        unassigned_loss = assignment_mask * ((codebook - jax.lax.stop_gradient(features_rev)) ** 2).mean(axis=-1)
        unassigned_loss = unassigned_loss.sum() / jnp.maximum(assignment_mask.sum(), 1)
        losses = dict(
            codebook=codebook_loss,
            commitment=commitment_loss,
            unassigned=unassigned_loss,
            unassigned_percent=(assignment_count > 0).mean()
        )
        return out_features, assign_fwd, losses

class FSQ(hk.Module):
    """Finite scalar quantization module.
    
    Not used in the manuscript.
    """
    def __init__(self, name: str | None = "fsq"):
        super().__init__(name)

    def __call__(self, features):
        downcast = Linear(7, bias=False)(features)
        half_l = (3 * (1 - 1e-3)) / 2
        offset = 0.5
        shift = jnp.tan(offset / half_l)
        downcast = jnp.tanh(downcast + shift) * half_l - offset
        rounded = jnp.round(jax.lax.stop_gradient(downcast))
        rounded = jax.lax.stop_gradient(rounded) + (downcast - jax.lax.stop_gradient(downcast))
        upcast = Linear(features.shape[-1], bias=False)(rounded / 2)
        return upcast, rounded

class VQState(hk.Module):
    """Vector quantization module with state."""
    def __init__(self, codebook_size=4096, gamma=0.99,
                 mapped_axes=None,
                 name: str | None = "vq"):
        super().__init__(name)
        self.codebook_size = codebook_size
        self.gamma = gamma
        self.mapped_axes = mapped_axes if mapped_axes else []

    def __call__(self, features, mask):
        def replace_gradient(x, y):
            return jax.lax.stop_gradient(x) + y - jax.lax.stop_gradient(y)
        codebook = hk.get_state("codebook",
                                (self.codebook_size, features.shape[-1]),
                                init=hk.initializers.RandomNormal())
        prev_codebook = codebook
        count = hk.get_state("count",
                             (self.codebook_size,),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(0.0))
        avg = hk.get_state("avg",
                             (self.codebook_size,),
                             dtype=jnp.float32,
                             init=hk.initializers.Constant(0.0))
        distance = ((features[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
        distance = jnp.where(mask[:, None], distance, jnp.inf)
        assign_fwd = jnp.argmin(distance, axis=1)
        assign_rev = jnp.argmin(distance, axis=0)
        features_fwd = codebook[assign_fwd]
        features_rev = features[assign_rev]
        out_features = replace_gradient(features_fwd, features)
        total = jnp.maximum(mask.sum(), 1)
        codebook_loss = ((features_fwd - jax.lax.stop_gradient(features)) ** 2).mean(axis=-1)
        codebook_loss = jnp.where(mask, codebook_loss, 0).sum() / total
        commitment_loss = ((jax.lax.stop_gradient(features_fwd) - features) ** 2).mean(axis=-1)
        commitment_loss = jnp.where(mask, commitment_loss, 0).sum() / total
        assignment_count = jnp.zeros((self.codebook_size,),
                                     dtype=jnp.float32).at[assign_fwd].add(mask)
        # if we are mapping over one or more axes
        # sum assignment_count over all of them
        if not hk.running_init():
            for axis in self.mapped_axes:
                assignment_count = jax.lax.psum(assignment_count, axis)
        assignment_mask = assignment_count < 1
        count = (1 - self.gamma) * assignment_count + self.gamma * count
        avg = (1 - self.gamma) * assignment_count / total  + self.gamma * avg
        alpha = jnp.exp(-avg * codebook.shape[0] * 10 / (1 - self.gamma) - 1e-3)
        assigned_update = self.gamma * codebook + (1 - self.gamma) * jnp.zeros_like(codebook).at[assign_fwd].add(features)
        assigned_update /= jnp.maximum(count[:, None], 1)
        unassigned_update = (1 - alpha)[:, None] * codebook + alpha[:, None] * features_rev
        codebook = jnp.where(assignment_mask[:, None], assigned_update, unassigned_update)
        # update codebook with average update over replicas
        codebook_delta = prev_codebook - codebook
        if not hk.running_init():
            for axis in self.mapped_axes:
                codebook_delta = jax.lax.pmean(codebook_delta, axis)
        codebook = prev_codebook + codebook_delta
        losses = dict(
            commitment=commitment_loss,
            unassigned_percent=(assignment_count > 0).mean()
        )
        state_update = dict(count=count, avg=avg, codebook=codebook)
        return out_features, assign_fwd, losses, state_update

class QuickDistogram(hk.Module):
    """Per-layer distogram module.
    
    Not used in the manuscript (too slow to be viable).
    """
    def __init__(self, config, name: Optional[str] = "quick_distogram"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, resi, chain, batch, neighbours):
        dcode_left = Linear(32, bias=False)(features)
        dcode_right = Linear(32, bias=False)(features)
        dcode = dcode_left[:, None] + dcode_right[neighbours]
        pair_resi = sequence_relative_position(32, one_hot=True)(
            resi, chain, batch, neighbours)
        dcode += Linear(32, bias=False)(pair_resi)
        dcode = hk.LayerNorm([-1], True, True)(dcode)
        distogram_logits = MLP(64, 16, depth=2, activation=jax.nn.gelu, final_init="zeros")(dcode)
        distogram_logits = jax.nn.log_softmax(distogram_logits, axis=-1)
        start = 0.0
        stop = 22.0
        step = (stop - start) / 16
        bin_centers = jnp.arange(16) * step + step / 2
        dmap = (jax.nn.softmax(distogram_logits, axis=-1) * bin_centers).sum(axis=-1)
        return distogram_logits, dmap

class InnerDistogram(hk.Module):
    """Light-weight per-layer distogram module. (+dist in the manuscript)"""
    def __init__(self, config, name: Optional[str] = "inner_distogram"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, resi, chain, batch, neighbours):
        dcode = Linear(16 * 8, bias=False)(features).reshape(features.shape[0], 8, 16)
        dgate = Linear(16 * 8, bias=False)(features).reshape(features.shape[0], 8, 16)
        dgate = jax.nn.gelu(dgate)
        weight = hk.get_parameter("inner_weight", (8, 8, 16), init=hk.initializers.Constant(0.0))
        logits = jnp.einsum("iax,jbx,abx->ijx", dcode, dgate, weight)
        logits = (logits + jnp.swapaxes(logits, 0, 1)) / 2
        logits = jax.nn.log_softmax(logits, axis=-1)
        start = 0.0
        stop = 22.0
        step = (stop - start) / 16
        bin_centers = jnp.arange(16) * step + step / 2
        dmap = (jax.nn.softmax(logits, axis=-1) * bin_centers).sum(axis=-1)
        return logits, dmap

class NonEquivariantDecoderBlock(hk.Module):
    """Fully non-equivariant decoder block.
    
    Not used in the manuscript.
    """
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features,
                 resi, chain, batch, mask,
                 sup_neighbours=None,
                 pos_gt=None):
        c = self.config
        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None
        }[c.distogram_block]
        dist = distogram_block(c)
        distogram_logits, dmap = dist(features, resi, chain, batch, None)
        index = axis_index(sup_neighbours, axis=0)
        distogram_logits = distogram_logits[index[:, None], sup_neighbours]
        dmap_neighbours = extract_dmap_neighbours(
            count=32)(
                jax.lax.stop_gradient(dmap),
                resi, chain, batch, mask)

        pair, pair_mask = nonequivariant_decoder_pair_features(c)(
            hk.LayerNorm([-1], True, True)(features),
            dmap, dmap_neighbours,
            resi, chain, batch, mask)
        features += SparseAttention(size=c.key_size,
                                    heads=c.heads)(
            hk.LayerNorm([-1], True, True)(features),
            pair, dmap_neighbours, pair_mask)
        # global update of local features
        features += NonEquivariantDecoderUpdate(c)(
            hk.LayerNorm([-1], True, True)(features),
            chain, batch, mask)
        return features, distogram_logits

class SemiEquivariantDecoderBlock(hk.Module):
    """Partly non-equivariant decoder block (+dgram in the manuscript)."""
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 resi, chain, batch, mask,
                 sup_neighbours=None,
                 pos_gt=None):
        c = self.config
        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None
        }[c.distogram_block]
        dist = distogram_block(c)
        distogram_logits, dmap = dist(features, resi, chain, batch, None)
        index = axis_index(sup_neighbours, axis=0)
        distogram_logits = distogram_logits[index[:, None], sup_neighbours]
        dmap_neighbours = extract_dmap_neighbours(
            count=32)(
                jax.lax.stop_gradient(dmap),
                resi, chain, batch, mask)
        nr = int(getattr(c, "num_random_neighbours", 32))
        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=nr)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)

        pair, pair_mask = hybrid_decoder_pair_features(c, center_features=False)(
            pos, dmap, current_neighbours,
            resi, chain, batch, mask)
        features += SparseSemiEquivariantPointAttention(c.key_size, c.heads)(
            hk.LayerNorm([-1], True, True)(features),
            pair, pos / c.sigma_data,
            current_neighbours, pair_mask)
        if distogram_block is not None:
            # FIXME: was decoder pair features
            pair, pair_mask = hybrid_decoder_pair_features(c)(
                pos, dmap, dmap_neighbours,
                resi, chain, batch, mask)
            features += SparseSemiEquivariantPointAttention(c.key_size, c.heads)(
                hk.LayerNorm([-1], True, True)(features),
                pair, pos / c.sigma_data,
                dmap_neighbours, pair_mask)
        # global update of local features
        features += NonEquivariantDecoderUpdate(c)(
            hk.LayerNorm([-1], True, True)(features),
            chain, batch, mask)
        local_norm = hk.LayerNorm([-1], True, True)(features)
        # update positions
        split_scale_out = c.sigma_data
        pos = semiequivariant_update_positions(pos, local_norm,
                                               scale=split_scale_out,
                                               symm=c.symm)
        return features, pos.astype(local_norm.dtype), distogram_logits

class DecoderBlock(hk.Module):
    """Standard equivariant decoder block."""
    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos,
                 resi, chain, batch, mask,
                 sup_neighbours=None,
                 pos_gt=None):
        c = self.config
        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None
        }[c.distogram_block]
        if distogram_block is not None:
            dist = distogram_block(c)
            if c.teacher_forcing_style:
                distogram_logits, _ = dist(features, resi, chain, batch, sup_neighbours)
                cb = Vec3Array.from_array(pos_gt[:, -1])
                dmap = (cb[:, None] - cb[None, :]).norm()
            else:
                distogram_logits, dmap = dist(features, resi, chain, batch, None)
                index = axis_index(sup_neighbours, axis=0)
                distogram_logits = distogram_logits[index[:, None], sup_neighbours]
            dmap_neighbours = extract_dmap_neighbours(
                count=32)(
                    jax.lax.stop_gradient(dmap),
                    resi, chain, batch, mask)
        else:
            # placeholder
            distogram_logits = jnp.zeros((1,), dtype=jnp.float32)
            dmap = None
        nr = int(getattr(c, "num_random_neighbours", 32))
        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=nr)(
                Vec3Array.from_array(pos),
                resi, chain, batch, mask)

        pair, pair_mask = decoder_pair_features(c)(
            pos, dmap, current_neighbours,
            resi, chain, batch, mask)
        features += SparseStructureAttention(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos / c.sigma_data, pair, pair_mask,
                current_neighbours, resi, chain, batch, mask)
        if distogram_block is not None:
            pair, pair_mask = decoder_pair_features(c)(
                pos, dmap, dmap_neighbours,
                resi, chain, batch, mask)
            features += SparseStructureAttention(c)(
                    hk.LayerNorm([-1], True, True)(features),
                    pos / c.sigma_data, pair, pair_mask,
                    dmap_neighbours, resi, chain, batch, mask)
        # global update of local features
        features += DecoderUpdate(c)(
                hk.LayerNorm([-1], True, True)(features),
                pos, chain, batch, mask)
        local_norm = hk.LayerNorm([-1], True, True)(features)
        # update positions
        split_scale_out = c.sigma_data
        pos = update_positions(pos, local_norm,
                               scale=split_scale_out,
                               symm=c.symm)
        return features, pos.astype(local_norm.dtype), distogram_logits


class DecoderBlockAtom37(hk.Module):
    """Equivariant decoder block for the parallel atom37 branch."""
    def __init__(self, config, name: Optional[str] = "decoder_block_atom37"):
        super().__init__(name)
        self.config = config

    def __call__(self, features, pos37,
                 resi, chain, batch, mask):
        c = self.config
        nr = int(getattr(c, "num_random_neighbours", 32))
        current_neighbours = extract_neighbours(
            num_index=16,
            num_spatial=16,
            num_random=nr)(
                Vec3Array.from_array(pos37),
                resi, chain, batch, mask)
        # Use a lightweight dmap derived from the atom37 branch itself.
        cb37 = Vec3Array.from_array(atom37_to_ncacocb(pos37)[:, -1])
        dmap37 = (cb37[:, None] - cb37[None, :]).norm()
        dmap_neighbours37 = extract_dmap_neighbours(
            count=32)(
                jax.lax.stop_gradient(dmap37),
                resi, chain, batch, mask)

        pair, pair_mask = decoder_pair_features_atom37(c)(
            pos37, dmap37, current_neighbours,
            resi, chain, batch, mask)
        features += SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos37 / c.sigma_data, pair, pair_mask,
            current_neighbours, resi, chain, batch, mask)

        pair, pair_mask = decoder_pair_features_atom37(c)(
            pos37, dmap37, dmap_neighbours37,
            resi, chain, batch, mask)
        features += SparseStructureAttention(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos37 / c.sigma_data, pair, pair_mask,
            dmap_neighbours37, resi, chain, batch, mask)

        features += DecoderUpdateAtom37(c)(
            hk.LayerNorm([-1], True, True)(features),
            pos37, chain, batch, mask)
        local_norm = hk.LayerNorm([-1], True, True)(features)
        pos37 = update_positions_atom37(
            pos37, local_norm,
            scale=c.sigma_data,
            symm=c.symm)
        return features, pos37.astype(local_norm.dtype)


def make_pair_mask(mask, neighbours):
    return mask[:, None] * mask[neighbours] * (neighbours != -1)

class EncoderUpdate(hk.Module):
    """GeGLU update for the encoder."""
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, chain, batch, mask):
        c = self.config
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        local_local = local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(local_local)
        return result

class DecoderUpdate(hk.Module):
    """GeGLU update with global averaging for the decoder."""
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos, chain, batch, mask):
        c = self.config
        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_array().reshape(local_pos.shape[0], -1)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden += index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden += local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result


class EncoderUpdateAtom37(hk.Module):
    """GeGLU update for the parallel atom37 encoder branch."""
    def __init__(self, config, name: Optional[str] = "light_global_update_atom37"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos37, chain, batch, mask):
        c = self.config
        _, local_pos37 = extract_aa_frames_atom37(Vec3Array.from_array(pos37))
        local_pos37 = local_pos37.to_array().reshape(local_pos37.shape[0], -1)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos37)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        local_local = local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(local_local)
        return result


class DecoderUpdateAtom37(hk.Module):
    """GeGLU update with global averaging for the parallel atom37 decoder."""
    def __init__(self, config, name: Optional[str] = "light_global_update_atom37"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, pos37, chain, batch, mask):
        c = self.config
        _, local_pos37 = extract_aa_frames_atom37(Vec3Array.from_array(pos37))
        local_pos37 = local_pos37.to_array().reshape(local_pos37.shape[0], -1)
        local += MLP(local.shape[-1] * 2,
                     local.shape[-1],
                     activation=jax.nn.gelu,
                     final_init=init_zeros())(local_pos37)
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden += index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden += local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result


class NonEquivariantDecoderUpdate(hk.Module):
    """Fully non-equivariant update for the decoder.
    
    Not used in the manuscript
    """
    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__(name)
        self.config = config

    def __call__(self, local, chain, batch, mask):
        c = self.config
        local_update = Linear(local.shape[-1] * c.factor, initializer=init_linear(), bias=False)(local)
        local_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        chain_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        batch_gate = jax.nn.gelu(Linear(local.shape[-1] * c.factor, initializer=init_relu(), bias=False)(local))
        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden += index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden += local_gate * local_update
        result = Linear(local.shape[-1], initializer=init_zeros())(hidden)
        return result

def update_positions(pos, local_norm, scale=10.0, symm=None):
    """Equivariant position update."""
    frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
    pos_update = scale * Linear(
        pos.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    # potentially symmetrize position update
    if symm is not None:
        pos_update = symm(pos_update)
    pos_update = Vec3Array.from_array(
        pos_update.reshape(*pos_update.shape[:-1], -1, 3))
    local_pos += pos_update

    # project updated pos to global coordinates
    pos = frames[..., None].apply_to_point(local_pos).to_array()
    return pos


def update_positions_atom37(pos37, local_norm, scale=10.0, symm=None):
    """Equivariant position update for full atom37 coordinates."""
    frames37, local_pos37 = extract_aa_frames_atom37(Vec3Array.from_array(pos37))
    pos_update37 = scale * Linear(
        pos37.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    if symm is not None:
        pos_update37 = symm(pos_update37)
    pos_update37 = Vec3Array.from_array(
        pos_update37.reshape(*pos_update37.shape[:-1], -1, 3))
    local_pos37 += pos_update37
    pos37 = frames37[..., None].apply_to_point(local_pos37).to_array()
    return pos37


def semiequivariant_update_positions(pos, local_norm, scale=10.0, symm=None):
    """Partially equivariant position update. (NEQ)"""
    pos_update = scale * Linear(
        pos.shape[-2] * 3, initializer=init_zeros(),
        bias=False)(local_norm)
    # potentially symmetrize position update
    pos_update = pos_update.reshape(*pos_update.shape[:-1], -1, 3)
    if symm is not None:
        pos_update = symm(pos_update)

    # project updated pos to global coordinates
    pos += pos_update
    return pos
