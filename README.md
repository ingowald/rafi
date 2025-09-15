# rafi - the "RAy Forwarding Infrastructure" Library

*DISCLAIMER* : this is VERY(!) "rough-and-ready" code that I wouldn't
usually call ready to share, yet; the only reason this repo isn't
private is that I need to share this code with collaborators; at this
point you might think twice (or more times!) before acutally trying to
use that code - though if you did stumble across this and you do have
an applicatoin where this would be useful I'd love to hear about it
fore sure..

basic ideas: mpi-based infrastructure library where app can define
what a 'ray' is, and how rays get 'forwarded' between different ranks.
Basically, each rank can generate a wave-front of 'rays', and then
have (CUDA-)kernels that process such wave-fronts. For each ray
processed in a wavefront the cuda kernel can then go back to rafi and
tell it to foward this ray to another rank - and rafi will make sure
that for the next iteration that ray will be on that other rank.

Key ideas:

- use as a submodule, app can then use this to forward 'rays' (or
  simiarl stuff) using templates.

- rafi offers two interfaces: one to the host-side app itself, and one
  to the CUDA kernel operating on said rasy.

Host interface:

- app initialized rafi with MPI communicator, and max number of rays
  per rank (this is a guarantee by the app to make sure never to have
  more than this number of rays on any rank; it's up to the app to
  guarantee that).
  
- app generates a new wavefront of rays on each rank, then operates in
  a sequence of alternating between app-specific _processing_ of a
  rank's current set of rays (which can, for each ray, specify a next
  rank that this ray needs to go to), and _forwarding_ the rays to
  wherever the processing stage decided they need to go.
  
Device Interface:

- app can query a 'device interface' it can pass to a cuda kernel

- cuda kenrel can ask for number of rays, and pointer to current rays

- for each ray, cuda kernel can say 'rafi.forwardRay(int
  nextRank)'. Rays that do ont get forwarded die after this wave,
  otherwise the ray gets sent to the indicated ray.
