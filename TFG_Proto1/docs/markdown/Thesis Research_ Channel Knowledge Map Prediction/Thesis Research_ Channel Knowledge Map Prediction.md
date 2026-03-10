# **Channel Knowledge Map Prediction with Deep Learning for 6G UAV-Enabled Networks**

## **Introduction to Environment-Aware 6G Communications**

The deployment of sixth-generation (6G) wireless networks introduces a fundamental paradigm shift toward environment-aware communications. Unlike preceding generations that largely treated the physical propagation environment as a passive, uncontrollable, and fundamentally adversarial medium, 6G architectures actively integrate the geometrical, structural, and material realities of the surrounding environment into network optimization and signal processing frameworks. <sup>1</sup> This transition relies heavily on the integration of Non-Terrestrial Networks (NTNs), specifically utilizing Uncrewed Aerial Vehicles (UAVs) to establish dynamic, heterogeneous, and ultra-massive connectivity grids. <sup>1</sup> The incorporation of UAVs as aerial base stations, mobile relays, and high-altitude terminals allows for unprecedented flexibility in providing line-of-sight (LoS) dominated links, alleviating the terrestrial blockage probabilities that severely degrade sub-6 GHz and millimeter-wave (mmWave) communications. 4

Concurrent with this architectural evolution is a strategic, industry-wide pivot toward the FR3 upper-midband spectrum, spanning from 7.125 GHz to 24.25 GHz. <sup>6</sup> The 7.125 GHz band, defined as a primary test band within the European Union and global standardization bodies, serves as a critical bridge. 1 It inherits the broad coverage capabilities of legacy sub-6 GHz frequencies while approaching the extreme capacity, wide bandwidth, and high-resolution sensing potential of mmWave bands. <sup>6</sup> However, the FR3 band introduces unique physical propagation characteristics, including extended near-field regions, higher free-space path loss than 5G mid-bands, and spatially nonstationary fading that complicates traditional stochastic channel modeling. 6

To optimize these advanced 6G networks without incurring the prohibitive computational overhead and latency of continuous, real-time Channel State Information (CSI) acquisition, the concept of the Channel Knowledge Map (CKM) has emerged. <sup>2</sup> A CKM is a highly localized, site-specific database tagged with the three-dimensional locations of transmitters and receivers, containing proactive, environment-aware channel parameters. <sup>2</sup> By visualizing and storing complex radio frequency environment states, CKMs obviate the need for sophisticated, energy-intensive real-time channel estimation protocols, serving as a cornerstone for 6G network digital twins. 2

Traditionally, CKMs and similar Radio Environment Maps (REMs) are generated using

deterministic Ray Tracing (RT) software, which simulates the physical interactions of radio waves with urban geometry. <sup>1</sup> While RT yields exceptional physical accuracy, its computational complexity makes it entirely incompatible with the real-time, highly dynamic demands of UAV-enabled networks where nodes operate at varying altitudes and high velocities. <sup>1</sup> To resolve this critical bottleneck, the research community has pivoted toward data-driven artificial intelligence, particularly advanced deep learning architectures capable of predicting channel parameters directly from 3D environmental geometry. <sup>1</sup> By framing 3D channel prediction as an advanced image-to-tensor translation problem, modern neural networks are being developed to map physical environments directly to complex radio channel characteristics with unprecedented speed and scalability. 1

### **The 6G FR3 Spectrum and UAV Network Dynamics**

Understanding the necessity for deep learning-driven CKMs requires a thorough analysis of the physical and spatial dynamics characterizing 6G UAV networks operating in the FR3 band. The transition to the 7.125 GHz frequency fundamentally alters the interaction between electromagnetic waves and urban topology. 1

#### **Propagation Characteristics of the 7.125 GHz Band**

The FR3 spectrum seeks to balance coverage, capacity, and deployment challenges. <sup>7</sup> Unlike the sub-6 GHz spectrum, which enjoys robust penetration through building materials and extensive diffraction around urban obstacles, the 7.125 GHz band suffers from more pronounced attenuation, higher free-space path loss, and sparser multipath components (MPCs). 7 Conversely, it avoids the extreme blockage susceptibility and severe atmospheric absorption that plague the mmWave (FR2) and sub-terahertz bands. 7

However, the shorter wavelength at 7.125 GHz (approximately 4.2 centimeters) means that urban surfaces appear physically rougher to the incident radio waves, leading to complex scattering phenomena that are difficult to capture using generalized stochastic models. 6 Furthermore, the deployment of Extremely Large Aperture Arrays (ELAAs) and ultra-massive MIMO systems in the FR3 band significantly extends the Rayleigh distance, placing many users and interacting scatterers within the radiative near-field region of the antenna. 6 In this near-field regime, the traditional plane-wave assumption collapses, and spherical wavefronts must be modeled, necessitating unified near-field and far-field channel prediction models capable of capturing spatially nonstationary fading. 6

#### **Aerial Node Dynamics and 3D Spatial Complexity**

UAV-enabled networks introduce a continuous, dynamic variation in antenna height that is largely absent in terrestrial cellular networks. <sup>1</sup> UAVs may operate at varying altitudes, ranging from low-level street canopy flights (e.g., 10 to 40 meters) to higher-altitude deployments

reaching up to approximately 500 meters. 1

This vertical mobility completely invalidates 2D planar channel models. As a UAV ascends or descends, the specific incidence angles of electromagnetic waves interacting with building rooftops, facades, and terrestrial scatterers change continuously. <sup>17</sup> A link that is heavily shadowed by a 20-meter building when the UAV is at a 15-meter altitude may suddenly transition to a pristine line-of-sight link when the UAV ascends to 25 meters. <sup>18</sup> Consequently, channel modeling for UAVs must account for elevation angular spreads, vertically structured shadowing, and the exact Z-axis deployment height of the antenna, rendering the prediction problem intrinsically three-dimensional. 1

## **The Computational Bottleneck of Deterministic Ray Tracing**

For decades, the gold standard for site-specific channel modeling has been deterministic Ray Tracing (RT). RT software algorithmically launches discrete rays from a transmitter and calculates their paths through a digital 3D environment, applying the laws of geometric optics to simulate reflection, diffraction, penetration, and diffuse scattering. <sup>16</sup> By incorporating the principles of physics, deterministic models can accurately calculate the precise path loss, phase, delay, and angle of each reflected component reaching the receiver. 12

Despite its physical fidelity, RT suffers from a crippling computational burden. The complexity of a standard RT algorithm scales roughly on the order of , where represents the

number of receiver locations and represents the number of ray interactions (reflections/diffractions) modeled. 13 In a high-density metropolitan area represented by complex multi-polygon meshes, running high-accuracy physical simulations—such as Intelligent Ray Tracing with up to four interactions (IRT4)—requires solving millions of intersection equations. <sup>21</sup> Research indicates that computing a single high-resolution RT map for a standard urban grid can take hundreds to thousands of seconds. 21

For highly dynamic 3D networks involving mobile UAVs, running RT simulations for every possible spatial configuration creates a severe, insurmountable bottleneck. 1 If a UAV modifies its altitude or trajectory, the entire channel matrix must be recalculated. This latency fundamentally precludes practical scalability and is entirely incompatible with the millisecond-level, real-time demands of 6G predictive precoding, beam alignment, and handover management. 1

#### **Acceleration Techniques and Their Limitations**

To mitigate this bottleneck, intermediate engineering solutions have proposed pre-processing

and post-processing acceleration techniques for RT pipelines. 24

- 1. **Pre-processing:** This involves simplifying the 3D scenario by executing polygon reduction algorithms on the environmental meshes. By reducing the geometric complexity of the digital city, the number of ray-surface intersection tests is lowered. 20
- 2. **Post-processing:** This entails augmenting a sparse set of reference RT results to achieve an improved time resolution, interpolating channels between discrete simulated frames. 20

While these optimizations theoretically reduce RT simulation times by more than 50% without severely compromising the accuracy of multipath parameters (such as delay, phase, and path gain), they remain insufficient for online operation. <sup>11</sup> A 50% reduction of a 100-second simulation still yields a 50-second latency, which is orders of magnitude too slow for autonomous UAV control loops or 6G integrated sensing and communication (ISAC) tasks. 6 Consequently, the industry focus has shifted toward standalone machine learning architectures capable of bypassing explicit ray calculation entirely. 1

# **Target Predictive Parameters for 6G UAV Networks**

To fully replace RT, a deep learning architecture must predict a comprehensive suite of channel characteristics. Predicting mere signal strength is inadequate for 6G; the network must infer the complex spatio-temporal properties of the channel from raw geometric inputs. <sup>1</sup> For a 6G FR3 UAV network, the target parameters are strictly defined with aggressive accuracy thresholds to ensure operational viability.

| Channel<br>Parameter                  | Description<br>and<br>Physical<br>Relevance<br>for<br>6G<br>UAVs                                                                                                                                                                   | Deep<br>Learning<br>Target<br>Accuracy<br>(RMSE) |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| Path<br>Loss<br>/<br>Channel<br>Power | Large-scale<br>macroscopic<br>attenuation<br>over<br>distance<br>and<br>through<br>obstacles.<br>Critical<br>for<br>defining<br>cell<br>coverage<br>boundaries,<br>basic<br>link<br>budgets,<br>and<br>interference<br>mitigation. | 1<br>3<br>to<br>5<br>dB                          |
| Delay<br>Spread<br>(RMS-DS)           | Temporal<br>dispersion<br>of<br>the<br>signal<br>due<br>to<br>multipath<br>reflections<br>arriving<br>at<br>different<br>times.<br>Dictates<br>inter-symbol<br>interference<br>and<br>maximum<br>achievable                        | 1<br>50<br>nanoseconds                           |

|                                  | bandwidth.                                                                                                                                                                                                       |                                                                                                 |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| Angular<br>Spread                | Spatial<br>dispersion<br>(azimuth<br>and<br>elevation)<br>of<br>multipath<br>components.<br>Essential<br>for<br>optimizing<br>6G<br>ELAA<br>massive<br>MIMO<br>beamforming<br>and<br>spatial<br>multiplexing.    | 1<br>20<br>degrees                                                                              |
| Augmented<br>Line<br>of<br>Sight | Classification<br>of<br>semi-obstructed<br>paths<br>(e.g.,<br>foliage,<br>partial<br>canopy).<br>Allows<br>trajectory<br>optimization<br>without<br>requiring<br>pure,<br>unobstructed<br>optical<br>visibility. | Binary/Classification<br>accuracy<br>(e.g.,<br>Probability<br>1<br>of<br>Detection<br>><br>90%) |

#### **Path Loss and Channel Power**

Path Loss (PL) represents the reduction in power density of an electromagnetic wave as it propagates through space. For deep learning models, the objective is to map 3D urban geometries directly to spatial PL distributions, targeting an RMSE of 3 to 5 dB. <sup>1</sup> This target is highly competitive; traditional stochastic models often exhibit errors exceeding 8–10 dB in complex urban environments because they fail to capture site-specific shadowing. 12

Recent UAV-assisted measurement campaigns utilizing sliding correlation and constant false alarm rate (CFAR) approaches to extract sparse channel knowledge have validated the feasibility of mapping aerial PL. <sup>5</sup> When evaluating UAV transmissions at 3.6 GHz across campus and farmland scenarios at varying altitudes (10m to 40m), models achieved PL RMSEs ranging from -16.34 dB to -17.54 dB against normalized ground truth data. <sup>5</sup> For 7.125 GHz 6G networks, maintaining tight error bounds across an entire city map requires the neural network to accurately infer both free-space decay and material penetration losses directly from visual representations. 1

## **Delay Spread**

Root Mean Square Delay Spread (RMS-DS) quantifies the standard deviation of the time delays of multiple signal paths. In an urban canyon, a signal transmitted from a UAV will bounce off various building facades and the ground, reaching the receiver at staggered intervals. <sup>8</sup> This causes frequency-selective fading and inter-symbol interference. For 6G services demanding

extreme data rates, predicting delay spread with an error margin of 50 nanoseconds or less is required. 1

Predicting RMS-DS via neural networks involves extracting topological features from building geometries. <sup>28</sup> The neural network implicitly learns that a receiver located deep within a narrow urban street will experience a dense cluster of reflections (high delay spread), whereas a receiver positioned on a rooftop with a clear line of sight to the UAV will experience a dominant single path with minimal delay spread. <sup>10</sup> Advanced preprocessing steps, such as normalizing continuous path delays based on the minimum and maximum delays in the dataset ( ), enable the network to better bound and predict temporal dispersion across the spatial grid. 30

#### **Angular Spread**

Angular Spread (AS) characterizes the variance in the Angle of Arrival (AoA) and Angle of Departure (AoD) of multipath components. In 6G UAV networks employing multi-antenna arrays, knowing the azimuth and elevation angular spread is critical for determining the optimal beamwidth for transmission. 28 If the AS is wide, a highly directional, pencil-thin beam might miss significant multipath energy; if the AS is narrow, a wider beam wastes transmit power and causes inter-cell interference. <sup>23</sup> Deep learning architectures target an AS prediction RMSE of no more than 20 degrees. 1

Machine learning tackles angular prediction by structuring the channel sampling over angular domains using Discrete Fourier Transform (DFT) matrices, which act to sparsify the complex channel data. <sup>30</sup> Neural architectures then map multi-dimensional inputs to these spatial bins, predicting the concentration of energy in specific angular sectors. Predicting elevation AS is particularly critical for UAVs, as the varying flight height directly manipulates the vertical angle of the dominant propagation paths interacting with terrestrial users. 29

## **Augmented Line of Sight (LoS)**

Traditional wireless models classify links as either strictly Line of Sight (LoS) or Non-Line of Sight (NLoS). However, physical environments—particularly suburban, rural, or agricultural landscapes—are rarely this binary. <sup>27</sup> A UAV transmitting to ground sensors may experience a signal path that passes through tree canopies, low-stalk crops, or minor structural overhangs. This scenario is defined as "Augmented Line of Sight". 27

In deep learning CKM predictions, an Augmented LoS metric is treated as an output tensor that applies an additional, deterministically predicted attenuation parameter overlaid onto the standard free-space path loss model. <sup>1</sup> Rather than treating a tree as a solid impenetrable block (which would erroneously plunge the prediction into severe NLoS fading), the model learns the semi-permeable nature of specific environments. By outputting a combined tensor that includes an Augmented LoS map, UAV trajectory optimization algorithms can route drones

through corridors that maintain computationally tractable connectivity without strictly requiring absolute optical clearance. 1

## **3D Environmental Datasets and Tensor Formatting**

The predictive accuracy of environment-aware deep learning is fundamentally tethered to the quality, resolution, scale, and specific formatting of the underlying 3D spatial data. 1 Transitioning from simple 2D near-ground modeling to 3D aerial UAV mapping requires exhaustive structural datasets and carefully engineered input tensors. 1

#### **GlobalBuildingAtlas and Urban Topology**

To ensure that deep learning architectures generalize effectively across diverse global urban layouts rather than overfitting to a single simulated city, massive topographical datasets are required. <sup>1</sup> The introduction of the GlobalBuildingAtlas represents a monumental advancement for wireless machine learning. 35

The GlobalBuildingAtlas provides the first open, globally complete high-resolution dataset of Level of Detail 1 (LoD1) 3D building models, containing over 2.75 billion building instances. 35 Achieving a horizontal spatial resolution of 3m x 3m—which is 30 times finer than previous 90m global products—it provides reliable building polygons and explicit building heights extracted via machine learning from global satellite imagery. <sup>35</sup> For CKM prediction, this repository supplies the raw geographical data necessary to simulate millions of diverse urban, suburban, and rural environments, ensuring that neural networks can be trained on everything from dense skyscrapers to sparse residential layouts. 1

#### **RadioMap3DSeer and UrbanRadio3D**

Bridging the gap between raw physical geometry and radio frequency ground truth are specialized datasets designed specifically for training radio map estimators.

| Dataset      | Environmental<br>Scope<br>&<br>Resolution                       | Simulation<br>Parameters                                                           | Key<br>Features<br>for<br>UAV<br>Networks                                                                                |
|--------------|-----------------------------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| RadioMapSeer | 701<br>City<br>Maps,<br>256x256m<br>grids,<br>1m<br>resolution. | IRT2<br>and<br>IRT4<br>(Intelligent<br>Ray<br>Tracing),<br>DPM.<br>56,000<br>maps. | Standard<br>baseline<br>for<br>2D<br>models.<br>Evaluates<br>effects<br>of<br>cars<br>and<br>missing<br>21<br>buildings. |

| RadioMap3DSeer | 701<br>City<br>Maps,<br>256x256m<br>grids.     | IRT2<br>(max<br>2<br>ray<br>interactions),<br>3.5<br>GHz,<br>20<br>MHz<br>bandwidth.       | Transmitters<br>explicitly<br>deployed<br>3<br>meters<br>above<br>rooftops.<br>Building<br>heights<br>range<br>from<br>6.6m<br>to<br>19.8m.<br>Explicitly<br>models<br>elevation<br>18<br>differences.      |
|----------------|------------------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| UrbanRadio3D   | High-fidelity<br>urban<br>volumetric<br>grids. | Ray<br>tracing<br>across<br>20<br>horizontal<br>slices<br>from<br>ground<br>to<br>rooftop. | 1-meter<br>cubic<br>resolution<br>providing<br>true<br>3D<br>x<br>3D<br>channel<br>representation.<br>Captures<br>volumetric<br>Pathloss,<br>Angle<br>of<br>Arrival,<br>and<br>17<br>Time<br>of<br>Arrival. |

UrbanRadio3D is particularly revolutionary for aerial networks, as it discards the assumption of a fixed receiver height. <sup>17</sup> By conducting ray-tracing simulations across 20 different height levels per urban scene, the dataset yields a complete 3D spatial distribution of wireless characteristics. <sup>17</sup> This allows generative models to learn volumetric propagation patterns intrinsic to high-rise and UAV networks, modeling how signals cascade vertically down building faces. 17

### **Input Tensor Construction**

To interface these 3D urban models with CNNs, GANs, or Diffusion models, raw geographic data must be structured into mathematically tractable tensors. <sup>1</sup> A highly optimized input methodology for 6G UAV CKM prediction frames the problem as an image-to-tensor translation task, incorporating the following channels 1 :

- 1. **Grayscale Image Matrices:** A 2D matrix representing the physical building footprints, where pixel values indicate the presence of structural boundaries. 1
- 2. **Height Matrices:** To convey 3D information without the memory overhead of voxel grids, a grayscale matrix is used where pixel intensity explicitly encodes the Z-axis height of the structures (e.g., a pixel value of 1 represents ground level, while a value of 255 represents the highest building in the domain). 1
- 3. **UAV Transmitter State Tensor:** A separate input matrix indicating the spatial coordinates of the antenna (usually centered in the image for symmetric processing), alongside a numerical encoding of the UAV's specific Z-value (altitude). 1

4. **Binary Line of Sight Maps (Optional):** Pre-computed matrices classifying explicit optical LoS boundaries. Injecting this deterministic, easily calculated geometric data directly into the input tensor accelerates network convergence by allowing the model to focus its learning capacity on complex diffraction and NLoS fading rather than basic geometry. 1

The output of the neural architecture is a corresponding set of tensors—one per image frame—containing the continuously varying values for Delay Spread, Angular Spread, Channel Power, and Augmented LoS. 1

# **Deep Learning Architectures for 3D Radio Map Generation**

With the datasets established and the tensors formatted, the core of the predictive engine relies on advanced deep learning software architectures. <sup>1</sup> The evolution of these models traces a path from basic convolutional networks to highly complex generative systems capable of synthesizing hyper-realistic radio environments.

| Architecture<br>Paradigm                                                   | Core<br>Mechanism                                                                                                                      | Primary<br>Advantages                                                                                                           | Limitations                                                                                                                                                     |
|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Convolutional<br>Neural<br>Networks<br>(e.g.,<br>RadioUNet)                | Encoder-decoder<br>structure<br>mapping<br>spatial<br>correlations<br>via<br>2D/3D<br>convolutional<br>14<br>filters.                  | Exceptional<br>execution<br>speed<br>(<br>sec).<br>Strong<br>baseline<br>accuracy<br>for<br>path<br>21<br>loss.                 | Often<br>produces<br>blurred<br>outputs;<br>struggles<br>with<br>highly<br>multi-dimensional<br>variance<br>(e.g.,<br>simultaneous<br>AoA<br>17<br>and<br>ToA). |
| Generative<br>Adversarial<br>Networks<br>(e.g.,<br>RME-GAN,<br>RadioGen3D) | Minimax<br>game<br>between<br>a<br>Generator<br>synthesizing<br>maps<br>and<br>a<br>Discriminator<br>evaluating<br>16<br>authenticity. | Synthesizes<br>realistic<br>shadowing<br>and<br>localized<br>deep<br>fading.<br>Excellent<br>for<br>data<br>16<br>augmentation. | Susceptible<br>to<br>mode<br>collapse;<br>requires<br>careful<br>tuning<br>of<br>41<br>adversarial<br>loss.                                                     |
| Diffusion<br>Models<br>(e.g.,<br>RadioDiff-3D)                             | Iterative<br>denoising<br>process<br>conditioned<br>on                                                                                 | State-of-the-art<br>accuracy<br>for<br>multi-feature<br>3D                                                                      | Computationally<br>heavier<br>inference<br>times<br>compared<br>to                                                                                              |

| environmental<br>17<br>geometry. | volumes.<br>Captures<br>high-frequency<br>spatial<br>dynamics<br>17<br>perfectly. | single-pass<br>CNNs. |
|----------------------------------|-----------------------------------------------------------------------------------|----------------------|
|----------------------------------|-----------------------------------------------------------------------------------|----------------------|

### **Convolutional Neural Networks and RadioUNet**

Early environment-aware frameworks leveraged Convolutional Neural Networks (CNNs) to extract spatial propagation patterns, bypassing the explicit classification of environmental materials. <sup>14</sup> The seminal architecture in this domain is RadioUNet, which treats radio map estimation strictly as a supervised image-to-image mapping problem. 22

RadioUNet utilizes a dual-path UNet architecture containing an contracting path to capture context (building geometries) and a symmetric expanding path that enables precise localization of signal strengths. <sup>21</sup> The framework features several specialized configurations:

- **RadioUNetC:** Operates purely on city maps and transmitter locations, functioning as a direct surrogate for physical simulation tools. It generates estimations exceptionally fast, taking only to seconds to compute a 256x256m map. 21
- **RadioUNetS:** Accepts sparse, real-world pathloss measurements alongside the city map. This hybrid approach allows the model to "infill" predictions and adapt to inaccuracies in the underlying map data, proving highly robust in real-world deployments. 21
- **3D-Adapted RadioUNet:** Instead of a single 2D image, this variant ingests 12 discrete horizontal slices of the building environment, effectively perceiving vertical shadow lengths behind buildings of varying heights. Benchmarks indicate this 3D-awareness reduces the RMSE from 1.26 dB (naive 2D approach) to 0.87 dB. 21

While RadioUNet provides a rapid, accurate baseline, traditional 2D convolution kernels lack the mathematical capacity to capture the complex volumetric propagation patterns and multipath angular dispersions inherent to UAV aerial networks. 17

### **Conditional Generative Adversarial Networks (cGANs)**

To elevate the structural fidelity of the predicted maps and accurately represent the sharp discontinuities caused by structural shadowing, the field transitioned toward Generative Adversarial Networks. 40 In a conditional GAN (cGAN), the generation process is heavily constrained by specific input features—namely, the topographical height matrices and the UAV spatial coordinates—ensuring a deterministic, one-to-one mapping rather than random noise generation. 39

Architectures like RME-GAN and RadioGen3D embed a 3D U-Net as the generator backbone within an adversarial training loop. <sup>16</sup> The generator attempts to synthesize a multi-tensor CKM containing pathloss, delay spread, and angular spread, while the discriminator evaluates the

generated tensors against ground-truth ray-tracing data. <sup>16</sup> By applying different sampling strategies to extract global path loss patterns alongside localized shadowing features, these models yield superior estimations of high-frequency fading components, producing outputs that closely align with the underlying statistical distributions of the radio channel. 40

### **Diffusion Models and Volumetric Synthesis**

Representing the current state-of-the-art in predictive modeling, diffusion models have been adapted to manage the overwhelming complexity of 3D spatial dependencies. <sup>17</sup> Frameworks such as RadioDiff and RadioDiff-3D formulate CKM construction as a conditional generative process governed by iterative denoising. 17

RadioDiff-3D employs 3D convolutional operators embedded within its diffusion backbone to synthesize high-fidelity radio maps across full spatial volumes. <sup>17</sup> To manage computational efficiency, these models often leverage a decoupled diffusion process combined with a Variational Autoencoder (VAE) encoder for compact latent representation. <sup>42</sup> Furthermore, an adaptive Fast Fourier Transform (FFT) module is integrated to enhance the model's ability to extract and reproduce the high-frequency spatial variations resulting from dynamic environmental features. <sup>48</sup> By modeling joint spatial dependencies throughout all three dimensions, RadioDiff-3D effectively bridges the gap between fixed-height terrestrial maps and the continuous volumetric spaces required by 6G UAVs. 17

# **Physics-Informed Neural Networks and Heuristic Constraints**

A fundamental vulnerability of purely data-driven deep learning models—whether CNNs or Generative models—is that they operate as black boxes. <sup>3</sup> They lack inherent physical interpretability and can fail catastrophically when applied to out-of-distribution environments, such as a city layout entirely unrepresented in their training data. <sup>50</sup> To guarantee reliability and generalization across diverse global layouts, 6G CKM prediction requires the integration of domain knowledge, establishing Physics-Informed Neural Networks (PINNs). 3

PINNs constrain the neural network's optimization landscape by mathematically embedding physical wave propagation laws directly into the loss function. 52 Instead of relying solely on minimizing the Mean Absolute Error (MAE) or RMSE against the dataset, the network minimizes a composite objective function. 12

The Total Loss Function ( ) in a state-of-the-art PINN architecture takes the form:

$$L_{total} = L_{data} + \lambda_1 L_{physics} + \lambda_2 L_{rea}$$

- 1. **Data-Driven Loss ( ):** Calculates the standard prediction error (e.g., RMSE) between the predicted channel parameters and the ground-truth RT data. 12
- 2. **Physics-Driven Loss ( ):** Penalizes outputs that violate established electromagnetic relationships. 54
- 3. **Regularization Loss ( ):** Prevents overfitting and regularizes network complexity. 54

# **Embedding the Laws of Physics**

Several distinct physical heuristics are integrated into the term to guide the network 1 :

- **The Helmholtz Equation:** Used in models like the Radio Map Diffusion Model (RMDM), this second-order partial differential equation describes the behavior of electromagnetic waves. <sup>52</sup> By enforcing the Helmholtz equation within the neural network's loss function, the model learns the precise mechanics of wave interference and diffraction around building corners, rather than just statistical correlations. 52
- **Laplacian Spatial Consistency:** To model the structural shadowing of the environment, networks utilize the Laplacian operator ( ) of the predicted power. <sup>12</sup> This forces the network to ensure that signal decay rates realistically conform to 3D obstacle occlusion, guaranteeing that sudden drops in signal strength correspond logically to the presence of physical walls in the height matrix. 12
- **Ray Optics and Sparsity:** For predicting Delay Spread and Angular Spread, PINNs leverage the laws of ray optics and Sparse Bayesian Generative Modeling (SBGM). <sup>50</sup> This imposes a sparsity-inducing characteristic on the network, aligning with the physical reality that mmWave and FR3 channels are naturally sparse in the angular domain. 30 It prevents the network from erroneously predicting a uniform scattering of multipath energy where discrete geometric reflections should exist. 50

## **Heuristic Feedback Control Layers**

To further stabilize training and ensure that the outputted data makes sense with known formulas for 6G UAV networks, advanced frameworks integrate a Heuristic Feedback Control Layer (HFL). 1 Initially developed for autonomous vehicle control, the HFL explicitly incorporates

current state errors into the forward pass of the network. <sup>26</sup> By formulating a gain matrix derived from the neural network's output, the HFL applies a deterministic correction factor during training. Methods utilizing heuristic feedback layers exhibit drastically faster convergence, smoother loss curves, and almost total elimination of steady-state prediction errors, maximizing the efficiency of limited ground-truth datasets while guaranteeing physical plausibility. 26

# **Generative Data Augmentation Pipelines**

A critical challenge spanning all CKM prediction architectures is data scarcity. <sup>34</sup> Generating a robust dataset of Intelligent Ray Tracing simulations for hundreds of complex 3D cities at varying UAV altitudes demands millions of hours of computational time. <sup>1</sup> To compensate for the computational expense of generating massive ground-truth datasets, advanced data augmentation pipelines must be implemented. 1

Traditional data augmentation techniques—such as simple image rotation, cropping, or adding uniform random noise to received signal strengths—are insufficient for radio maps, as they often violate the underlying physics of the environment. <sup>45</sup> Consequently, generative AI, specifically Generative Adversarial Networks (GANs), are repurposed as data augmentation engines. 45

#### **CGAN and WGAN-GP Augmentation**

By utilizing Conditional GANs (CGANs), researchers can expand partial training datasets by synthesizing entirely new, highly realistic channel data. <sup>34</sup> A CGAN trained on a small subset of accurate ray-tracing maps learns the complex non-linear mapping between building geometry, UAV height, and signal attenuation. <sup>58</sup> Once trained, the generator can output thousands of synthetic radio maps for entirely new, procedurally generated 3D city layouts. <sup>45</sup> Simulation results validate that this generated data exhibits high consistency with actual channels, accurately capturing the statistical properties in both the spatial and delay domains. 58

However, standard GANs often suffer from "mode collapse," a failure state where the generator produces a limited variety of repetitive outputs, failing to capture the full diversity of real-world environments. <sup>41</sup> To counteract this and ensure broad generalization, advanced augmentation pipelines utilize Wasserstein GANs with Gradient Penalty (WGAN-GP). <sup>41</sup> WGAN-GP replaces the standard binary cross-entropy loss with a Wasserstein distance metric, which provides smoother, more reliable gradients during adversarial training. 41

Empirical studies deploying WGAN-GP for radio map augmentation demonstrate profound improvements in downstream tasks. By matching the distribution of actual training data and augmenting datasets at a 1:1 ratio, deep learning models trained on WGAN-GP augmented data have shown decreases in average localization and prediction errors by up to 22.2%. 59

### **Physics-Informed Filtering**

To guarantee that the synthetic data generated by these networks remains immaculate, the augmentation pipeline is coupled with the previously defined physical heuristics. <sup>1</sup> A filtering mechanism applies deterministic formulas—such as the Friis transmission equation extended for basic occlusion—to the outputs of the WGAN-GP. <sup>1</sup> Any synthetic channel realization that violates baseline electromagnetic energy conservation or presents impossible angular

dispersion is discarded. <sup>54</sup> This hybridized data augmentation strategy guarantees that the primary CKM prediction models train on massive, highly diverse, and physically infallible datasets, ensuring seamless generalization across global urban landscapes without the prohibitive cost of endless physical simulation. 1

## **Conclusion**

The orchestration of 6G UAV-enabled networks within the FR3 7.125 GHz spectrum necessitates a departure from the reactive, computationally exhaustive channel estimation techniques of prior generations. The transition toward proactive, environment-aware Channel

Knowledge Maps is paramount, yet the complexity of deterministic ray tracing renders physical simulation impossible for dynamic, 3D aerial mobility.

Deep learning provides the definitive solution, reframing 3D volumetric channel prediction as an advanced image-to-tensor translation problem. By utilizing sophisticated architectural backbones—ranging from the extreme execution speeds of RadioUNet to the high-fidelity multi-dimensional synthesis of RadioDiff-3D and RME-GAN—neural networks successfully compress the physics of wave propagation into highly efficient latent spaces. The synthesis of massive structural datasets, like the GlobalBuildingAtlas and UrbanRadio3D, with meticulously structured input tensors ensures that models can accurately interpret the Z-axis variations inherent to UAV operations.

Furthermore, the integration of Physics-Informed Neural Networks and heuristic feedback layers guarantees that these models do not merely approximate statistical noise but strictly adhere to the fundamental equations of electromagnetism. Supported by WGAN-GP generative data augmentation pipelines, these architectures possess the capability to accurately predict channel power, delay spread, angular spread, and augmented line-of-sight across any global urban geometry. Ultimately, the successful deployment of deep learning-driven CKMs establishes the foundational environmental awareness required to unlock the extreme capacity, reliability, and scale of next-generation 6G NTN communications.

#### **Obras citadas**

- 1. project proposal and workplan.docx (5).pdf
- 2. Towards Environment-Aware 6G Communications via Channel Knowledge Map arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/pdf/2007.09332>
- 3. A Comprehensive Survey of Knowledge-Driven Deep Learning for Intelligent Wireless Network Optimization in 6G - IEEE Xplore, fecha de acceso: marzo 5, 2026, <https://ieeexplore.ieee.org/iel8/9739/5451756/11017513.pdf>
- 4. 6G Wireless Communications in 7-24 GHZ Band: Opportunities, Techniques, and Challenges - Scribd, fecha de acceso: marzo 5, 2026, <https://www.scribd.com/document/846979976/2310-06425v2>
- 5. Channel Knowledge Map Construction Based on a UAV-Assisted ..., fecha de

- acceso: marzo 5, 2026, <https://www.mdpi.com/2504-446X/8/5/191>
- 6. Frequency Range 3 for ISAC in 6G: Potentials and Challenges arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2506.18243v2>
- 7. 6G Wireless Communications in 7–24 GHz Band: Opportunities, Techniques, and Challenges - arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2310.06425v2>
- 8. Measurement-Based Prediction of mmWave Channel Parameters Using Deep Learning and Point Cloud - IEEE Xplore, fecha de acceso: marzo 5, 2026, <https://ieeexplore.ieee.org/iel8/8782711/10345397/10620622.pdf>
- 9. A General 3D Space-Time-Frequency Non-Stationary THz Channel Model for 6G Ultra-Massive MIMO Wireless Communication Systems | Request PDF - ResearchGate, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/350911651\\_A\\_General\\_3D\\_Space-Time](https://www.researchgate.net/publication/350911651_A_General_3D_Space-Time-Frequency_Non-Stationary_THz_Channel_Model_for_6G_Ultra-Massive_MIMO_Wireless_Communication_Systems) [-Frequency\\_Non-Stationary\\_THz\\_Channel\\_Model\\_for\\_6G\\_Ultra-Massive\\_MIMO\\_](https://www.researchgate.net/publication/350911651_A_General_3D_Space-Time-Frequency_Non-Stationary_THz_Channel_Model_for_6G_Ultra-Massive_MIMO_Wireless_Communication_Systems) [Wireless\\_Communication\\_Systems](https://www.researchgate.net/publication/350911651_A_General_3D_Space-Time-Frequency_Non-Stationary_THz_Channel_Model_for_6G_Ultra-Massive_MIMO_Wireless_Communication_Systems)
- 10. A Recent Survey on Radio Map Estimation Methods for Wireless Networks MDPI, fecha de acceso: marzo 5, 2026, <https://www.mdpi.com/2079-9292/14/8/1564>
- 11. Accelerating Ray Tracing-Based Wireless Channels Generation for Real-Time Network Digital Twins - ResearchGate, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/393038920\\_Accelerating\\_Ray\\_Tracing-](https://www.researchgate.net/publication/393038920_Accelerating_Ray_Tracing-Based_Wireless_Channels_Generation_for_Real-Time_Network_Digital_Twins)[Based\\_Wireless\\_Channels\\_Generation\\_for\\_Real-Time\\_Network\\_Digital\\_Twins](https://www.researchgate.net/publication/393038920_Accelerating_Ray_Tracing-Based_Wireless_Channels_Generation_for_Real-Time_Network_Digital_Twins)
- 12. ReVeal: A Physics-Informed Neural Network for High-Fidelity Radio ..., fecha de acceso: marzo 5, 2026,
  - <https://wici.iastate.edu/wp-content/uploads/2025/03/ReVeal-DySPAN25.pdf>
- 13. Wireless channel modeling and estimation by artificial neural networks RWTH Publications, fecha de acceso: marzo 5, 2026, <https://publications.rwth-aachen.de/record/820357/files/820357.pdf>
- 14. Radio Map Prediction from Aerial Images and Application to Coverage Optimization - IEEE Xplore, fecha de acceso: marzo 5, 2026, <https://ieeexplore.ieee.org/iel8/7693/4656680/11063460.pdf>
- 15. Machine Learning Based Radio Environment Maps for 4G/5G Networks ResearchGate, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/395775191\\_Machine\\_Learning\\_Based\\_R](https://www.researchgate.net/publication/395775191_Machine_Learning_Based_Radio_Environment_Maps_for_4G5G_Networks) [adio\\_Environment\\_Maps\\_for\\_4G5G\\_Networks](https://www.researchgate.net/publication/395775191_Machine_Learning_Based_Radio_Environment_Maps_for_4G5G_Networks)
- 16. RadioGen3D: 3D Radio Map Generation via Adversarial Learning on Large-Scale Synthetic Data - arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2602.18744v1>
- 17. RadioDiff-3D: A 3D×3D Radio Map Dataset and Generative Diffusion Based Benchmark for 6G Environment-Aware Communication - arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2507.12166v1>
- 18. Dataset of Pathloss and ToA Radio Maps With ... arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/abs/2212.11777>
- 19. RadioGen3D: 3D Radio Map Generation via Adversarial Learning OpenTrain AI, fecha de acceso: marzo 5, 2026,

- [https://www.opentrain.ai/papers/radiogen3d-3d-radio-map-generation-via-adver](https://www.opentrain.ai/papers/radiogen3d-3d-radio-map-generation-via-adversarial-learning-on-large-scale-synth--arxiv-2602.18744/) [sarial-learning-on-large-scale-synth--arxiv-2602.18744/](https://www.opentrain.ai/papers/radiogen3d-3d-radio-map-generation-via-adversarial-learning-on-large-scale-synth--arxiv-2602.18744/)
- 20. Accelerating Ray Tracing-Based Wireless Channels Generation for Real-Time Network Digital Twins - IEEE Xplore, fecha de acceso: marzo 5, 2026, <https://ieeexplore.ieee.org/iel8/8782661/10829557/11050957.pdf>
- 21. RadioUNet: Fast Radio Map Estimation with Convolutional Neural ..., fecha de acceso: marzo 5, 2026, <https://arxiv.org/abs/1911.09002>
- 22. (PDF) RadioUNet: Fast Radio Map Estimation With Convolutional Neural Networks, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/349280869\\_RadioUNet\\_Fast\\_Radio\\_Ma](https://www.researchgate.net/publication/349280869_RadioUNet_Fast_Radio_Map_Estimation_With_Convolutional_Neural_Networks) [p\\_Estimation\\_With\\_Convolutional\\_Neural\\_Networks](https://www.researchgate.net/publication/349280869_RadioUNet_Fast_Radio_Map_Estimation_With_Convolutional_Neural_Networks)
- 23. 10x the capacity with half the energy: Site-specific deep learning for 6G Asilomar Conference on Signals, Systems, and Computers, fecha de acceso: marzo 5, 2026, [https://www.asilomarsscconf.org/webpage/asil24/AndrewsPlenary\\_Asilomar2024.](https://www.asilomarsscconf.org/webpage/asil24/AndrewsPlenary_Asilomar2024.pdf) [pdf](https://www.asilomarsscconf.org/webpage/asil24/AndrewsPlenary_Asilomar2024.pdf)
- 24. Accelerating Ray Tracing-Based Wireless Channels Generation for Real-Time Network Digital Twins - arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2504.09751v2>
- 25. Accelerating Ray Tracing-Based Wireless Channels Generation for Real-Time Network Digital Twins - arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2504.09751v1>
- 26. Physics-informed Machine Learning with Heuristic Feedback ..., fecha de acceso: marzo 5, 2026, <https://www.merl.com/publications/docs/TR2025-087.pdf>
- 27. Transformer-Based Soft Actor–Critic for UAV Path Planning in Precision Agriculture IoT Networks - PMC, fecha de acceso: marzo 5, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC12737083/>
- 28. An Environment-Data-Physics Driven Model for 6G V2V Urban Channels IEEE Xplore, fecha de acceso: marzo 5, 2026, <https://ieeexplore.ieee.org/iel8/7693/11199987/10988553.pdf>
- 29. Deep Learning-Based Indoor Localization Using Multi-View BLE Signal PMC, fecha de acceso: marzo 5, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9003244/>
- 30. Predicting Wireless Channel Features using Neural Networks arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/pdf/1802.00107>
- 31. A Multi-Task Learning Model for Super Resolution of Wireless Channel Characteristics - CERES Research Repository, fecha de acceso: marzo 5, 2026, [https://dspace.lib.cranfield.ac.uk/bitstreams/8063f393-7ffb-45d9-946b-7ede2b5](https://dspace.lib.cranfield.ac.uk/bitstreams/8063f393-7ffb-45d9-946b-7ede2b5ac320/download) [ac320/download](https://dspace.lib.cranfield.ac.uk/bitstreams/8063f393-7ffb-45d9-946b-7ede2b5ac320/download)
- 32. Suborbital Science Program NASA Science Mission Directorate, fecha de acceso: marzo 5, 2026, [https://airbornescience.nasa.gov/sites/default/files/documents/SSP06Rpt\\_final.pdf](https://airbornescience.nasa.gov/sites/default/files/documents/SSP06Rpt_final.pdf)
- 33. Transformer-Based Soft Actor–Critic for UAV Path Planning in Precision Agriculture IoT Networks - ResearchGate, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/398463903\\_Transformer-Based\\_Soft\\_A](https://www.researchgate.net/publication/398463903_Transformer-Based_Soft_Actor-Critic_for_UAV_Path_Planning_in_Precision_Agriculture_IoT_Networks)

- [ctor-Critic\\_for\\_UAV\\_Path\\_Planning\\_in\\_Precision\\_Agriculture\\_IoT\\_Networks](https://www.researchgate.net/publication/398463903_Transformer-Based_Soft_Actor-Critic_for_UAV_Path_Planning_in_Precision_Agriculture_IoT_Networks)
- 34. Simulation-Enhanced Data Augmentation for Machine Learning Pathloss Prediction - arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2402.01969v1>
- 35. GlobalBuildingAtlas: an open global and complete dataset of building polygons, heights and LoD1 3D models - ESSD Copernicus, fecha de acceso: marzo 5, 2026, <https://essd.copernicus.org/articles/17/6647/2025/>
- 36. All the world's buildings available as 3D models for the first time TUM, fecha de acceso: marzo 5, 2026, [https://www.tum.de/en/news-and-events/all-news/press-releases/details/all-the](https://www.tum.de/en/news-and-events/all-news/press-releases/details/all-the-worlds-buildings-available-as-3d-models-for-the-first-time)[worlds-buildings-available-as-3d-models-for-the-first-time](https://www.tum.de/en/news-and-events/all-news/press-releases/details/all-the-worlds-buildings-available-as-3d-models-for-the-first-time)
- 37. RadioMapSeer Dataset, fecha de acceso: marzo 5, 2026, <https://radiomapseer.github.io/>
- 38. AIRMap: AI-Generated Radio Maps for Wireless Digital Twins This work was supported in part by VIAVI Solutions, Inc. - arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2511.05522v3>
- 39. Radio map construction based on generative adversarial networks with ACT blocks - IET Digital Library, fecha de acceso: marzo 5, 2026, <https://digital-library.theiet.org/doi/pdf/10.1049/cmu2.12846?download=true>
- 40. RME-GAN: A Learning Framework for Radio Map Estimation based on Conditional Generative Adversarial Network | Request PDF - ResearchGate, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/370912878\\_RME-GAN\\_A\\_Learning\\_Fra](https://www.researchgate.net/publication/370912878_RME-GAN_A_Learning_Framework_for_Radio_Map_Estimation_based_on_Conditional_Generative_Adversarial_Network) [mework\\_for\\_Radio\\_Map\\_Estimation\\_based\\_on\\_Conditional\\_Generative\\_Adversar](https://www.researchgate.net/publication/370912878_RME-GAN_A_Learning_Framework_for_Radio_Map_Estimation_based_on_Conditional_Generative_Adversarial_Network) [ial\\_Network](https://www.researchgate.net/publication/370912878_RME-GAN_A_Learning_Framework_for_Radio_Map_Estimation_based_on_Conditional_Generative_Adversarial_Network)
- 41. Generative Adversarial Network-Based Data Augmentation for Enhancing Wireless Physical Layer Authentication - MDPI, fecha de acceso: marzo 5, 2026, <https://www.mdpi.com/1424-8220/24/2/641>
- 42. This is the code for paper "RadioDiff- \$k^2\$ : Helmholtz Equation Informed Generative Diffusion Model for Multi-Path Aware Radio Map Construction", accepted by IEEE JSAC. - GitHub, fecha de acceso: marzo 5, 2026, <https://github.com/UNIC-Lab/RadioDiff-k>
- 43. RonLevie/RadioUNet: Convolutional neural network for estimating radio maps in urban environments - GitHub, fecha de acceso: marzo 5, 2026, <https://github.com/RonLevie/RadioUNet>
- 44. RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/pdf/1911.09002>
- 45. GAN+: Data Augmentation Method using Generative Adversarial Networks and Dirichlet for Indoor Localisation - CEUR-WS.org, fecha de acceso: marzo 5, 2026, <https://ceur-ws.org/Vol-3097/paper8.pdf>
- 46. RME-GAN: A Learning Framework for Radio Map Estimation based on Conditional Generative Adversarial Network - IEEE Xplore, fecha de acceso: marzo 5, 2026, <https://ieeexplore.ieee.org/ielaam/6488907/10269651/10130091-aam.pdf>
- 47. RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction - University of Waterloo, fecha de acceso: marzo 5,

- 2026, [https://uwaterloo.ca/scholar/sites/ca.scholar/files/sshen/files/wang2025radiodiff.p](https://uwaterloo.ca/scholar/sites/ca.scholar/files/sshen/files/wang2025radiodiff.pdf) [df](https://uwaterloo.ca/scholar/sites/ca.scholar/files/sshen/files/wang2025radiodiff.pdf)
- 48. RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction - arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2408.08593v1>
- 49. Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals - NeurIPS, fecha de acceso: marzo 5, 2026, <https://neurips.cc/virtual/2025/123184>
- 50. ICML Poster Physics-Informed Generative Modeling of Wireless Channels, fecha de acceso: marzo 5, 2026, <https://icml.cc/virtual/2025/poster/45899>
- 51. Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals - OpenReview, fecha de acceso: marzo 5, 2026, <https://openreview.net/pdf?id=r3plaU6DvW>
- 52. RMDM: Radio Map Diffusion Model with Physics Informed arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2501.19160v1>
- 53. Physics-Informed Artificial Intelligence for Adaptive Wireless Channel Modelling in Fifth-Generation (5G) Networks | International Journal of Research and Applied Technology (INJURATECH) - Universitas Komputer Indonesia, fecha de acceso: marzo 5, 2026, <https://ojs.unikom.ac.id/index.php/injuratech/article/view/18015>
- 54. Physics-Informed Artificial Intelligence for Adaptive Wireless Channel Modelling in Fifth-Generation (5G) Networks - Unikom, fecha de acceso: marzo 5, 2026, <https://ojs.unikom.ac.id/index.php/injuratech/article/download/18015/5707>
- 55. [2502.10137] Physics-Informed Generative Modeling of Wireless Channels arXiv.org, fecha de acceso: marzo 5, 2026, <https://arxiv.org/abs/2502.10137>
- 56. Physics-Informed Generative Modeling of Wireless Channels arXiv, fecha de acceso: marzo 5, 2026, <https://arxiv.org/html/2502.10137v1>
- 57. [2010.00178] Training Data Augmentation for Deep Learning Radio Frequency Systems, fecha de acceso: marzo 5, 2026, <https://arxiv.org/abs/2010.00178>
- 58. CGAN-Based Data Augmentation for Enhanced Channel Prediction in Massive MIMO under Subway Tunnels - ResearchGate, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/390579234\\_CGAN-Based\\_Data\\_Augm](https://www.researchgate.net/publication/390579234_CGAN-Based_Data_Augmentation_for_Enhanced_Channel_Prediction_in_Massive_MIMO_under_Subway_Tunnels) [entation\\_for\\_Enhanced\\_Channel\\_Prediction\\_in\\_Massive\\_MIMO\\_under\\_Subway\\_Tu](https://www.researchgate.net/publication/390579234_CGAN-Based_Data_Augmentation_for_Enhanced_Channel_Prediction_in_Massive_MIMO_under_Subway_Tunnels) [nnels](https://www.researchgate.net/publication/390579234_CGAN-Based_Data_Augmentation_for_Enhanced_Channel_Prediction_in_Massive_MIMO_under_Subway_Tunnels)
- 59. Data Augmentation using GANs for Deep Learning-based Localization Systems, fecha de acceso: marzo 5, 2026, [https://www.researchgate.net/publication/355921410\\_Data\\_Augmentation\\_using\\_](https://www.researchgate.net/publication/355921410_Data_Augmentation_using_GANs_for_Deep_Learning-based_Localization_Systems) [GANs\\_for\\_Deep\\_Learning-based\\_Localization\\_Systems](https://www.researchgate.net/publication/355921410_Data_Augmentation_using_GANs_for_Deep_Learning-based_Localization_Systems)