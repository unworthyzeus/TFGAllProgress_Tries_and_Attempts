# UAV CKM Differences From Papers

This note summarizes what the papers in the workspace suggest is different when Channel Knowledge Map (CKM) prediction is applied to UAV or aerial links instead of the more usual ground-only radio-map setting.

It is not a generic opinion note. The points below are grounded in the local paper conversions already stored in the repo.

## Main conclusion

The literature suggests that UAV-enabled CKM prediction is not just "the same radio-map problem with a taller antenna".

The main differences are:

- the problem becomes genuinely 3D, not just 2D
- altitude changes the propagation regime itself, not only the numerical values
- LoS becomes more dominant but also more geometry-dependent in a vertical sense
- directional and temporal channel features become more important, not only path loss
- model inputs should encode height, relative geometry, and often antenna orientation more explicitly
- many existing radio-map datasets and baselines are too simplified for UAV use because they assume fixed heights or ground-only links

## 1. UAV links turn CKM from a 2D problem into a 3D problem

One of the clearest messages in the papers is that conventional radio-map prediction is often built around a fixed-height 2D plane. That is acceptable for many ground-user settings, but it becomes insufficient for UAV links.

The CKM survey in [2309.07460v2 (1) (1).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2309.07460v2%20(1)%20(1)/2309.07460v2%20(1)%20(1).md#L254) states that the map input dimension grows when users are aerial: for BS-centric CKM, ground UEs typically need a 2D location, while aerial UEs require 3D location. For X2X settings, the dimension grows further because both Tx and Rx may move in 3D.

The review in [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L44) argues that fixed-height 2D path-loss prediction does not generalize to vertical or directional dimensions and is therefore of limited use for volumetric coverage planning, beamforming, and aerial navigation.

Practical implication for your project:

- a UAV-aware CKM should treat altitude as a first-class variable
- fixed-height image-to-image prediction is a weaker approximation than in terrestrial settings
- if height varies across samples, the model should see that height explicitly or be trained over height layers

## 2. Height changes propagation patterns qualitatively, not only quantitatively

The papers do not describe height as a small perturbation. They describe it as a regime-changing factor.

In [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L184), the authors note that when observation height exceeds surrounding building heights, obstacles may effectively disappear from the radio map because the receiver regains clear LoS. That means the map structure itself changes with altitude, not just the absolute path-loss level.

The same paper repeatedly emphasizes height-varying propagation, vertical non-stationarity, and elevation-sensitive effects in [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L83) and [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L91).

Practical implication for your project:

- a single 2D map plus a weak scalar conditioning may be enough for a first baseline, but it is a compressed representation of a truly volumetric phenomenon
- the higher the UAV altitude range and the more diverse the urban morphology, the more likely you need either multi-height training data or stronger geometry-aware features

## 3. LoS is more important, but it is not just a binary floor-plan feature

For UAV and air-to-ground links, LoS tends to matter even more than in street-level terrestrial prediction. But the papers also make clear that the useful notion of LoS is more vertical and geometry-aware than a plain 2D binary mask.

The survey in [2309.07460v2 (1) (1).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2309.07460v2%20(1)%20(1)/2309.07460v2%20(1)%20(1).md#L764) explicitly criticizes free-space or probabilistic LoS UAV models for failing to capture site-specific A2G LoS availability.

The radio-map paper in [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L165) goes further and argues that simple binary Tx/object tensors are not enough when heights matter. It introduces relative cylindrical and spherical encodings, explicit elevation angles, and richer LoS descriptors in [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L185) and [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L209).

Practical implication for your project:

- for UAV settings, LoS should ideally depend on height and geometry relative to the antenna, not only on ground obstruction masks
- a true 3D or altitude-aware LoS feature is more defensible than a plain ground-level LoS bitmap
- your current use of `los_mask` as an input prior is useful, but the papers suggest it is still a simplified proxy compared with full A2G visibility structure

## 4. Directional and temporal channel descriptors matter more in aerial settings

A repeated criticism in the literature is that most public radio-map work only predicts path loss. For UAV and other 3D applications, this is not enough.

The 3D radio-map paper in [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L83) argues that path-loss-only maps miss critical descriptors such as DoA, ToA, delay spread, and angular spread, which are needed for beamforming, localization, interference coordination, and altitude-aware communication.

The same paper frames 3D RM as a multi-modal tensor over path loss, DoA, and ToA in [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L248). That is directly aligned with your thesis plan, which already prioritizes delay spread and angular spread in addition to the dB-domain map.

Practical implication for your project:

- your choice to predict `delay_spread` and `angular_spread` is more UAV-relevant than a path-loss-only baseline
- if future data allows it, elevation-related angular information or richer delay/arrival structure would be especially valuable for aerial use cases

## 5. Relative geometry matters more than raw images alone

The papers suggest that once Tx height, UAV height, and directional propagation become important, raw geometry images are often not the best sole representation.

In [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L165), the authors show that performance benefits from encoding information relative to the transmitter: distance to Tx, azimuth, relative object heights, elevation-related quantities, and antenna pattern projections. They also discuss explicit LoS-related encoding in [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L209).

This is especially relevant for UAVs because altitude changes whether an object blocks, reflects, or is irrelevant, and that is easier to infer from relative geometry than from a plain grayscale height map alone.

Practical implication for your project:

- a strong UAV-oriented model should probably know at least some of: antenna height, relative height to obstacles, distance to antenna, and possibly azimuth/elevation geometry
- your current fixed-center setup already helps because it standardizes the geometry around the antenna location
- but if you later move from fixed-center/fixed-height assumptions to real varying UAV altitude, richer relative encodings become more important

## 6. Antenna orientation and pattern become more consequential

In ground-only simplified datasets, isotropic assumptions are common. The papers argue this is a weak approximation for more realistic 3D settings.

The review in [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L64) explicitly criticizes public datasets that assume isotropic antennas and notes that this becomes problematic for higher-frequency bands. The same paper studies explicit antenna gain/orientation encodings in [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L185).

Practical implication for your project:

- if UAV deployment eventually includes directional aerial antennas, the model should not rely only on terrain/building geometry
- antenna orientation and pattern may need to become explicit inputs
- this is especially relevant in FR3 and beyond, where beam direction matters more

## 7. UAV-related radio maps can be easier in some settings, but that depends on the exact task

One useful warning from the literature is that not every UAV radio-map task is equally hard.

The review in [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L56) says that some works on Tx mounted on UAVs may actually be easier because the most complex between-building propagation does not arise to the same extent.

That does not contradict the other papers. It means the difficulty depends on the scenario:

- a high UAV above rooftops with mostly clear A2G links can be simpler than deep urban canyon ground propagation
- but a realistic 6G UAV-enabled CKM with varying heights, 3D mobility, beam control, and richer outputs is more complex overall than classic fixed-height 2D path-loss prediction

Practical implication for your project:

- if your current dataset keeps the antenna at a fixed center and geometry frame, part of the hardest mobility problem is still absent
- once you introduce varying UAV altitude or different aerial placements, the problem becomes more representative and harder

## 8. Dataset design becomes much more critical

The papers repeatedly argue that UAV-relevant CKM quality is often limited more by the dataset than by the choice between two CNN variants.

The main bottlenecks repeatedly mentioned are:

- fixed-height training data cannot teach vertical generalization
- 2D-only labels miss altitude-sensitive effects
- path-loss-only datasets miss delay and angular structure
- coarse altitude sampling is often insufficient
- lack of realistic building heights, vegetation, or antenna assumptions reduces transfer value

This is emphasized in [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L83), [2507.12166v1.md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2507.12166v1/2507.12166v1.md#L91), and [2402.00878v1 (2).md](c:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2402.00878v1%20(2)/2402.00878v1%20(2).md#L64).

Practical implication for your project:

- the current HDF5 route is a good operational baseline, but it is still a compressed version of the full UAV problem described in the proposal
- if the thesis later shifts toward stronger UAV realism, the biggest upgrade may be data representation rather than only model depth

## 9. What this means for your current repo specifically

Relative to the papers, your current pipeline already moves in the right direction in some ways:

- it predicts more than just path loss
- it includes `delay_spread` and `angular_spread`
- it uses building-height-derived input geometry
- it keeps the antenna-centered spatial frame explicit

But compared with the UAV-specific needs highlighted in the papers, the current route is still simplified in these ways:

- the HDF5 route is still effectively 2D per sample, not a full 3D volumetric map
- antenna height is not stored as per-sample metadata in the current HDF5 file
- the current LoS handling is simpler than the richer height-aware LoS encodings discussed in the papers
- directional antenna effects are not yet modeled as rich explicit inputs in the HDF5 path
- the dataset does not directly expose a multi-height stack for each scene in the way the 3D RM papers recommend

## Bottom line

According to the papers in this workspace, the main differences introduced by UAV or high-altitude antennas are:

1. The map domain becomes inherently 3D.
2. Altitude changes the propagation structure, not just the magnitude.
3. LoS must be modeled in a more height-aware way.
4. Path loss alone is less sufficient; delay, angle, and arrival structure matter more.
5. Relative geometry and antenna orientation become more important inputs.
6. Fixed-height 2D datasets and models are weak surrogates for the full UAV-enabled problem.

So the literature supports treating UAV CKM prediction as a more geometry-aware, altitude-aware, and often more multimodal problem than conventional terrestrial radio-map prediction.