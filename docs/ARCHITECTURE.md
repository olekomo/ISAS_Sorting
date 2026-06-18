# Architecture

## Dependency direction

```text
BenchMARL
  └── ISASClass
       └── ISASTorchEnv (TorchRL EnvBase)
            └── ISASDynamics
                 ├── ActorDynamics
                 ├── ParticleDynamics
                 ├── ContactEngine
                 ├── RewardEngine
                 └── ObservationEncoder

Rendering
  ├── RenderSnapshot  <- explicit copy of one environment only
  ├── MatplotlibRenderer
  └── FrameRecorder
```

The simulation core does not import BenchMARL, TensorDict, PettingZoo,
Matplotlib or image/video libraries. This separation is intentional:

- `ISASDynamics` can be unit tested directly;
- TorchRL owns reset/step/spec conventions;
- BenchMARL owns collection and algorithm selection;
- rendering can run after training, in evaluation, or in a separate process.

## State lifecycle

Every episode owns `P=num_particles` slots. A slot transitions as follows:

```text
unspawned -> active -> resolved
                       ├── hit
                       ├── disturbed
                       └── end_of_line
```

Slots are filled monotonically. `particle_outcome` is zero while unresolved and
uses `1=hit`, `2=disturbed`, `3=end_of_line` afterward.

## Random streams

Two device-local `torch.Generator` objects retain the legacy conceptual split:

- prediction generator: process noise during movement;
- initialization generator: initial states and per-step birth counts.

The initialization seed uses the previous offset `456789123`. A fixed seed and
fixed batch/reset sequence are reproducible. Bit-identical NumPy trajectories
are not expected because PyTorch uses a different PRNG implementation.

## Births

Each process samples a rounded non-negative normal birth count. When a final
step requests more particles than slots remain, static random token priorities
implement without-replacement capping across processes. Initial particle states
are pre-sampled at reset from the valid spawn region and copied into their slots
when born.

## Contacts

Contact tensors have shape `[B,A,P]`. A particle is:

- hit when it is inside an actor tolerance and the actor is HIT;
- directly disturbed in the same tolerance while the actor is UP or DOWN;
- disturbed when it previously entered an active disturbance rectangle and
  later leaves that rectangle without being hit by the same actor.

Disturbance wins over hit, matching the legacy contact logic.

## Partial reset

TorchRL may reset different batch members at different times. `_reset` accepts a
root `_reset` mask and replaces only selected rows of every state tensor.
Unselected environments retain their complete simulator state.
