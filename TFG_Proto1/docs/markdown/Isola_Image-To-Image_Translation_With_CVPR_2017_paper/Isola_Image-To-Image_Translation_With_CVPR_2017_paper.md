![](p1__page_0_Picture_1.jpeg)

# Image-to-Image Translation with Conditional Adversarial Networks

Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros

## Berkeley AI Research (BAIR) Laboratory, UC Berkeley

![](p1__page_0_Figure_5.jpeg)

<span id="page-0-0"></span>Figure 1: Many problems in image processing, graphics, and vision involve translating an input image into a corresponding output image. These problems are often treated with application-specific algorithms, even though the setting is always the same: map pixels to pixels. Conditional adversarial nets are a general-purpose solution that appears to work well on a wide variety of these problems. Here we show results of the method on several. In each case we use the same architecture and objective, and simply train on different data.

## Abstract

*We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Moreover, since the release of the* pix2pix *software associated with this paper, hundreds of twitter users have posted their own artistic experiments using our system. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without handengineering our loss functions either.*

### 1. Introduction

Many problems in image processing, computer graphics, and computer vision can be posed as "translating" an input image into a corresponding output image. Just as a concept may be expressed in either English or French, a scene may be rendered as an RGB image, a gradient field, an edge map, a semantic label map, etc. In analogy to automatic language translation, we define automatic *image-to-image translation* as the problem of translating one possible representation of a scene into another, given sufficient training data (see Figure [1\)](#page-0-0). Traditionally, each of these tasks has been tackled with separate, special-purpose machinery (e.g., [\[14,](#page-8-0) [23,](#page-8-1) [18,](#page-8-2) [8,](#page-8-3) [10,](#page-8-4) [50,](#page-9-0) [30,](#page-8-5) [36,](#page-8-6) [16,](#page-8-7) [55,](#page-9-1) [58\]](#page-9-2)), despite the fact that the setting is always the same: predict pixels from pixels. Our goal in this paper is to develop a common framework for all these problems.

The community has already taken significant steps in this direction, with convolutional neural nets (CNNs) becoming the common workhorse behind a wide variety of image prediction problems. CNNs learn to minimize a loss function – an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still goes into designing effective losses. In other words, we still have to tell the CNN what we wish it to minimize. But, just like King Midas, we must be careful what we wish for! If we take a naive approach, and ask the CNN to minimize Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results [\[40,](#page-9-3) [58\]](#page-9-2). This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring. Coming up with loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images – is an open problem and generally requires expert knowledge.

It would be highly desirable if we could instead specify only a high-level goal, like "make the output indistinguishable from reality", and then automatically learn a loss function appropriate for satisfying this goal. Fortunately, this is exactly what is done by the recently proposed Generative Adversarial Networks (GANs) [\[22,](#page-8-8) [12,](#page-8-9) [41,](#page-9-4) [49,](#page-9-5) [59\]](#page-9-6). GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss. Blurry images will not be tolerated since they look obviously fake. Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.

In this paper, we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model [\[22\]](#page-8-8). This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image.

GANs have been vigorously studied in the last two years and many of the techniques we explore in this paper have been previously proposed. Nonetheless, earlier papers have focused on specific applications, and it has remained unclear how effective image-conditional GANs can be as a general-purpose solution for image-toimage translation. Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results. Our second contribution is to present a simple framework sufficient to achieve good results, and to analyze the effects of several important architectural choices. Code is available at https://github.com/phillipi/pix2pix.

### 2. Related work

Structured losses for image modeling Image-to-image translation problems are often formulated as per-pixel classification or regression (e.g., [\[36,](#page-8-6) [55,](#page-9-1) [25,](#page-8-10) [32,](#page-8-11) [58\]](#page-9-2)). These formulations treat the output space as "unstructured" in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a *structured loss*. Structured losses penalize the joint configuration of the output. A large body of literature has considered losses of this kind, with methods including conditional random fields [\[9\]](#page-8-12), the SSIM metric [\[53\]](#page-9-7), feature matching [\[13\]](#page-8-13), nonparametric losses [\[34\]](#page-8-14), the convolutional pseudo-prior [\[54\]](#page-9-8), and losses based on matching covariance statistics [\[27\]](#page-8-15). The conditional GAN is different in that the loss is learned, and can, in theory, penalize any possible structure that differs between output and target.

Conditional GANs We are not the first to apply GANs in the conditional setting. Prior and concurrent works have conditioned GANs on discrete labels [\[38,](#page-8-16) [21,](#page-8-17) [12\]](#page-8-9), text [\[43\]](#page-9-9), and, indeed, images. The image-conditional models have tackled image prediction from a normal map [\[52\]](#page-9-10), future frame prediction [\[37\]](#page-8-18), product photo generation [\[56\]](#page-9-11), and image generation from sparse annotations [\[28,](#page-8-19) [45\]](#page-9-12) (c.f. [\[44\]](#page-9-13) for an autoregressive approach to the same problem). Several other papers have also used GANs for image-to-image mappings, but only applied the GAN unconditionally, relying on other terms (such as L2 regression) to force the output to be conditioned on the input. These papers have achieved impressive results on inpainting [\[40\]](#page-9-3), future state prediction [\[60\]](#page-9-14), image manipulation guided by user constraints [\[61\]](#page-9-15), style transfer [\[35\]](#page-8-20), and superresolution [\[33\]](#page-8-21). Each of the methods was tailored for a specific application. Our framework differs in that nothing is applicationspecific. This makes our setup considerably simpler than most others.

Our method also differs from the prior works in several architectural choices for the generator and discriminator. Unlike past work, for our generator we use a "U-Net"-based architecture [\[47\]](#page-9-16), and for our discriminator we use a convolutional "PatchGAN" classifier, which only penalizes structure at the scale of image patches. A similar PatchGAN architecture was previously proposed in [\[35\]](#page-8-20), for the purpose of capturing local style statistics. Here we show that this approach is effective on a wider range of problems, and we investigate the effect of changing the patch size.

## 3. Method

GANs are generative models that learn a mapping from random noise vector z to output image y, G : z → y [\[22\]](#page-8-8). In contrast, conditional GANs learn a mapping from observed image x and random noise vector z, to y, G : {x, z} → y. The generator G is trained to produce outputs that cannot be distinguished from "real" images by an adversarially trained discriminator, D, which is trained to do as well as possible at detecting the generator's "fakes". This training procedure is diagrammed in Figure [2.](#page-2-0)

### 3.1. Objective

The objective of a conditional GAN can be expressed as

$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x, z))], \quad (1)$$

where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. G<sup>∗</sup> = arg min<sup>G</sup> max<sup>D</sup> LcGAN (G, D).

<span id="page-2-0"></span>![](p1__page_2_Picture_0.jpeg)

Figure 2: Training a conditional GAN to map edges $\rightarrow$ photo. The discriminator, D, learns to classify between fake (synthesized by the generator) and real {edge, photo} tuples. The generator, G, learns to fool the discriminator. Unlike an unconditional GAN, both the generator and discriminator observe the input edge map.

To test the importance of conditioning the discriminator, we also compare to an unconditional variant in which the discriminator does not observe x:

$$\mathcal{L}_{GAN}(G, D) = \mathbb{E}_y[\log D(y)] + \mathbb{E}_{x,z}[\log(1 - D(G(x, z))]. \tag{2}$$

Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance [40]. The discriminator's job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:

$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\|y - G(x,z)\|_1]. \tag{3}$$

Our final objective is

$$G^* = \arg\min_{G} \max_{D} \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G).$$
 (4)

Without z, the net could still learn a mapping from xto y, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise z as an input to the generator, in addition to x (e.g., [52]). In initial experiments, we did not find this strategy effective - the generator simply learned to ignore the noise - which is consistent with Mathieu et al. [37]. Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time. Despite the dropout noise, we observe only minor stochasticity in the output of our nets. Designing conditional GANs that produce highly stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work.

#### 3.2. Network architectures

We adapt our generator and discriminator architectures from those in [41]. Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu [26]. Details of the architecture are provided in the supplemental materials online, with key features discussed below.

### 3.2.1 Generator with skips

A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution output grid. In addition, for the problems we consider, the input and output differ in surface appearance, but both are renderings of the same underlying structure. Therefore, structure in the input is roughly aligned with structure in the output. We design the generator architecture around these considerations.

Many previous solutions [40, 52, 27, 60, 56] to problems in this area have used an encoder-decoder network [24]. In such a network, the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed. Such a network requires that all information flow pass through all the layers, including the bottleneck. For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net. For example, in the case of image colorizaton, the input and output share the location of prominent edges.

To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a "U-Net" [47]. Specifically, we add skip connections between each layer i and layer n-i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer i.

#### <span id="page-2-1"></span>3.2.2 Markovian discriminator (PatchGAN)

It is well known that the L2 loss – and L1, see Figure 3 – produces blurry results on image generation problems [31]. Although these losses fail to encourage high-frequency crispness, in many cases they nonetheless accurately capture the low frequencies. For problems where this is the case, we do not need an entirely new framework to enforce correctness at the low frequencies. L1 will already do.

This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness (Eqn. 4). In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each  $N \times N$  patch in an image is real or fake. We run this discriminator convolutationally across the image, averaging all responses to provide the ultimate output of D.

In Section 4.4, we demonstrate that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller

PatchGAN has fewer parameters, runs faster, and can be applied on arbitrarily large images.

Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. This connection was previously explored in [35], and is also the common assumption in models of texture [15, 19] and style [14, 23, 20, 34]. Our PatchGAN can therefore be understood as a form of texture/style loss.

### 3.3. Optimization and inference

To optimize our networks, we follow the standard approach from [22]: we alternate between one gradient descent step on D, then one step on G. We use minibatch SGD and apply the Adam solver [29].

At inference time, we run the generator net in exactly the same manner as during the training phase. This differs from the usual protocol in that we apply dropout at test time, and we apply batch normalization [26] using the statistics of the test batch, rather than aggregated statistics of the training batch. This approach to batch normalization, when the batch size is set to 1, has been termed "instance normalization" and has been demonstrated to be effective at image generation tasks [51]. In our experiments, we use batch sizes between 1 and 10 depending on the experiment.

# 4. Experiments

To explore the generality of conditional GANs, we test the method on a variety of tasks and datasets, including both graphics tasks, like photo generation, and vision tasks, like semantic segmentation:

- Semantic labels↔photo, trained on the Cityscapes dataset [11]. Architectural labels→photo, trained on CMP Facades [42].
- Map⇔aerial photo, trained on data scraped from Google Maps.
- $BW \rightarrow color \ photos$ , trained on [48].
- Edges \rightarrow photo, trained on data from [61] and [57]; binary edges generated using the HED edge detector [55] plus postprocessing.
- tests edges-photo models on human-drawn
- sketches from [17].  $Day \rightarrow night$ , trained on [30].

Details of training on each of these datasets are provided in the supplemental materials online. In all cases, the input and output are simply 1-3 channel images. Qualitative results are shown in Figures 7, 8, 9, 10, and 11, with additional results and failure cases in the materials online (https://phillipi.github.io/pix2pix/).

#### 4.1. Evaluation metrics

Evaluating the quality of synthesized images is an open and difficult problem [49]. Traditional metrics such as perpixel mean-squared error do not assess joint statistics of the result, and therefore do not measure the very structure that structured losses aim to capture.

In order to more holistically evaluate the visual quality of our results, we employ two tactics. First, we run

<span id="page-3-0"></span>

| Loss         | Per-pixel acc. | Per-class acc. | Class IOU |
|--------------|----------------|----------------|-----------|
| L1           | 0.42           | 0.15           | 0.11      |
| GAN          | 0.22           | 0.05           | 0.01      |
| cGAN         | 0.57           | 0.22           | 0.16      |
| L1+GAN       | 0.64           | 0.20           | 0.15      |
| L1+cGAN      | 0.66           | 0.23           | 0.17      |
| Ground truth | 0.80           | 0.26           | 0.21      |

Table 1: FCN-scores for different losses, evaluated on Cityscapes labels↔photos.

"real vs fake" perceptual studies on Amazon Mechanical Turk (AMT). For graphics problems like colorization and photo generation, plausibility to a human observer is often the ultimate goal. Therefore, we test our map generation, aerial photo generation, and image colorization using this approach.

Second, we measure whether or not our synthesized cityscapes are realistic enough that off-the-shelf recognition system can recognize the objects in them. This metric is similar to the "inception score" from [49], the object detection evaluation in [52], and the "semantic interpretability" measures in [58] and [39].

**AMT perceptual studies** For our AMT experiments, we followed the protocol from [58]: Turkers were presented with a series of trials that pitted a "real" image against a "fake" image generated by our algorithm. On each trial, each image appeared for 1 second, after which the images disappeared and Turkers were given unlimited time to respond as to which was fake. The first 10 images of each session were practice and Turkers were given feedback. No feedback was provided on the 40 trials of the main experiment. Each session tested just one algorithm at a time, and Turkers were not allowed to complete more than one session.  $\sim 50$  Turkers evaluated each algorithm. All images were presented at  $256 \times 256$  resolution. Unlike [58], we did not include vigilance trials. For our colorization experiments, the real and fake images were generated from the same grayscale input. For map \( \to \) aerial photo, the real and fake images were not generated from the same input, in order to make the task more difficult and avoid floor-level results.

FCN-score While quantitative evaluation of generative models is known to be challenging, recent works [49, 52, 58, 39] have tried using pre-trained semantic classifiers to measure the discriminability of the generated stimuli as a pseudo-metric. The intuition is that if the generated images are realistic, classifiers trained on real images will be able to classify the synthesized image correctly as well. To this end, we adopt the popular FCN-8s [36] architecture for semantic segmentation, and train it on the cityscapes dataset. We then score synthesized photos by the classification accuracy against the labels these photos were synthesized from.

<span id="page-4-0"></span>![](p4__page_4_Figure_0.jpeg)

Figure 3: Different losses induce different quality of results. Each column shows results trained under a different loss. Please see https://phillipi.github.io/pix2pix/ for additional examples.

<span id="page-4-1"></span>![](p4__page_4_Figure_2.jpeg)

<span id="page-4-2"></span>Figure 4: Adding skip connections to an encoder-decoder to create a "U-Net" results in much higher quality results.

| Discriminator<br>receptive field | Per-pixel acc. | Per-class acc. | Class IOU |
|----------------------------------|----------------|----------------|-----------|
| 1×1                              | 0.39           | 0.15           | 0.10      |
| 16×16                            | 0.65           | 0.21           | 0.17      |
| 70×70                            | 0.66           | 0.23           | 0.17      |
| 286×286                          | 0.42           | 0.16           | 0.11      |

Table 2: FCN-scores for different receptive field sizes of the discriminator, evaluated on Cityscapes labels→photos. Note that input images are 256 × 256 pixels and larger receptive fields are padded with zeros.

# 4.2. Analysis of the objective function

Which components of the objective in Eqn. [4](#page-2-0) are important? We run ablation studies to isolate the effect of the L1 term, the GAN term, and to compare using a discriminator conditioned on the input (cGAN, Eqn. [1\)](#page-1-0) against using an unconditional discriminator (GAN, Eqn. [2\)](#page-2-1).

Figure [3](#page-4-0) shows the qualitative effects of these variations on two labels→photo problems. L1 alone leads to reasonable but blurry results. The cGAN alone (setting λ = 0 in Eqn. [4\)](#page-2-0) gives much sharper results, but introduces visual artifacts on certain applications. Adding both terms together (with λ = 100) reduces these artifacts.

We quantify these observations using the FCN-score on the cityscapes labels→photo task (Table [1\)](#page-3-0): the GAN-based objectives achieve higher scores, indicating that the synthesized images include more recognizable structure. We also test the effect of removing conditioning from the discriminator (labeled as GAN). In this case, the loss does not penalize mismatch between the input and output; it only cares that the output look realistic. This variant results in very poor performance; examining the results reveals that the generator collapsed into producing nearly the exact same output regardless of input photograph. Clearly it is important, in this case, that the loss measure the quality of the match between input and output, and indeed cGAN performs much better than GAN. Note, however, that adding an L1 term also encourages that the output respect the input, since the L1 loss penalizes the distance between ground truth outputs, which correctly match the input, and synthesized outputs, which may not. Correspondingly, L1+GAN is also effective at creating realistic renderings that respect the input label maps. Combining all terms, L1+cGAN, performs similarly well.

Colorfulness A striking effect of conditional GANs is that they produce sharp images, hallucinating spatial structure even where it does not exist in the input label map. One might imagine cGANs have a similar effect on "sharpening" in the spectral dimension – i.e. making images more colorful. Just as L1 will incentivize a blur when it is uncertain where exactly to locate an edge, it will also incentivize an average, grayish color when it is uncertain which of several plausible color values a pixel should take on. Specially, L1 will be minimized by choosing the median of of the conditional probability density function over possible colors. An adversarial loss, on the other hand, can in principle become aware that grayish outputs are unrealistic, and encourage matching the true color distribution [\[22\]](#page-8-7). In Figure [6,](#page-5-1) we investigate if our cGANs actually achieve this effect on the Cityscapes dataset. The plots show the marginal distributions over output color values in Lab color space. The ground truth distributions are shown with a dotted line. It is apparent that L1 leads to a narrower distribution than the ground truth, confirming the hypothesis that L1 encourages average, grayish colors. Using a cGAN, on the other hand, pushes the output distribution closer to the ground truth.

# 4.3. Analysis of the generator architecture

A U-Net architecture allows low-level information to shortcut across the network. Does this lead to better results? Figure [4](#page-4-1) compares the U-Net against an encoder-

<span id="page-5-3"></span>![](p4__page_5_Figure_0.jpeg)

Figure 5: Patch size variations. Uncertainty in the output manifests itself differently for different loss functions. Uncertain regions become blurry and desaturated under L1. The 1x1 PixelGAN encourages greater color diversity but has no effect on spatial statistics. The 16x16 PatchGAN creates locally sharp results, but also leads to tiling artifacts beyond the scale it can observe. The  $70\times70$  PatchGAN forces outputs that are sharp, even if incorrect, in both the spatial and spectral (colorfulness) dimensions. The full  $286\times286$  ImageGAN produces results that are visually similar to the  $70\times70$  PatchGAN, but somewhat lower quality according to our FCN-score metric (Table 2). Please see https://phillipi.github.io/pix2pix/ for additional examples.

<span id="page-5-1"></span>![](p4__page_5_Figure_2.jpeg)

Figure 6: Color distribution matching property of the cGAN, tested on Cityscapes. (c.f. Figure 1 of the original GAN paper [22]). Note that the histogram intersection scores are dominated by differences in the high probability region, which are imperceptible in the plots, which show log probability and therefore emphasize differences in the low probability regions.

<span id="page-5-0"></span>![](p4__page_5_Figure_4.jpeg)

Figure 7: Example results on Google Maps at 512x512 resolution (model was trained on images at  $256 \times 256$  resolution, and run convolutionally on the larger images at test time). Contrast adjusted for clarity.

decoder on cityscape generation. The encoder-decoder is created simply by severing the skip connections in the U-Net. The encoder-decoder is unable to learn to generate realistic images in our experiments. The advantages of the U-Net appear not to be specific to conditional GANs: when both U-Net and encoder-decoder are trained with an L1 loss, the U-Net again achieves the superior results (Figure 4).

### 4.4. From PixelGANs to PatchGans to ImageGANs

We test the effect of varying the patch size N of our discriminator receptive fields, from a  $1 \times 1$  "PixelGAN" to a full  $286 \times 286$  "ImageGAN". Figure 5 shows qualitative

results of this analysis and Table 2 quantifies the effects using the FCN-score. Note that elsewhere in this paper, unless specified, all experiments use  $70 \times 70$  PatchGANs, and for this section all experiments use an L1+cGAN loss.

The PixelGAN has no effect on spatial sharpness, but does increase the colorfulness of the results (quantified in Figure 6). For example, the bus in Figure 5 is painted gray when the net is trained with an L1 loss, but becomes red with the PixelGAN loss. Color histogram matching is a common problem in image processing [46], and PixelGANs may be a promising lightweight solution.

Using a  $16 \times 16$  PatchGAN is sufficient to promote sharp outputs, and achieves good FCN-scores, but also leads to

<span id="page-5-2"></span><sup>&</sup>lt;sup>1</sup>We achieve this variation in patch size by adjusting the depth of the GAN discriminator. Details of this process, and the discriminator architec-

tures are provided in the in the supplemental materials online.

<span id="page-6-4"></span>![](p7__page_6_Picture_0.jpeg)

Figure 8: Colorization results of conditional GANs versus the L2 regression from [\[58\]](#page-9-0) and the full method (classification with rebalancing) from [\[60\]](#page-9-1). The cGANs can produce compelling colorizations (first two rows), but have a common failure mode of producing a grayscale or desaturated result (last row).

<span id="page-6-0"></span>

|         | Photo → Map            | Map → Photo            |  |
|---------|------------------------|------------------------|--|
| Loss    | % Turkers labeled real | % Turkers labeled real |  |
| L1      | 2.8% ± 1.0%            | 0.8% ± 0.3%            |  |
| L1+cGAN | 6.1% ± 1.3%            | 18.9% ± 2.5%           |  |

<span id="page-6-1"></span>Table 3: AMT "real vs fake" test on maps↔aerial photos.

| Method                  | % Turkers labeled real |
|-------------------------|------------------------|
| L2 regression from [58] | 16.3% ± 2.4%           |
| Zhang et al. 2016 [58]  | 27.8% ± 2.7%           |
| Ours                    | 22.5% ± 1.6%           |

Table 4: AMT "real vs fake" test on colorization.

<span id="page-6-2"></span>

| Loss    | Per-pixel acc. | Per-class acc. | Class IOU |
|---------|----------------|----------------|-----------|
| L1      | 0.86           | 0.42           | 0.35      |
| cGAN    | 0.74           | 0.28           | 0.22      |
| L1+cGAN | 0.83           | 0.36           | 0.29      |

Table 5: Performance of photo→labels on cityscapes.

tiling artifacts. The 70 × 70 PatchGAN alleviates these artifacts and achieves similar scores. Scaling beyond this, to the full 286 × 286 ImageGAN, does not appear to improve the visual quality of the results, and in fact gets a considerably lower FCN-score (Table [2\)](#page-4-0). This may be because the ImageGAN has many more parameters and greater depth than the 70 × 70 PatchGAN, and may be harder to train.

Fully-convolutional translation An advantage of the PatchGAN is that a fixed-size patch discriminator can be applied to arbitrarily large images. We may also apply the generator convolutionally, on larger images than those on which it was trained. We test this on the map↔aerial photo task. After training a generator on 256×256 images, we test it on 512×512 images. The results in Figure [7](#page-5-0) demonstrate the effectiveness of this approach.

#### 4.5. Perceptual validation

We validate the perceptual realism of our results on the tasks of map↔aerial photograph and grayscale→color. Results of our AMT experiment for map↔photo are given in Table [3.](#page-6-0) The aerial photos generated by our method fooled participants on 18.9% of trials, significantly above the L1 baseline, which produces blurry results and nearly never fooled participants. In contrast, in the photo→map direction our method only fooled participants on 6.1% of trials, and this was not significantly different than the performance of the L1 baseline (based on bootstrap test). This may be because minor structural errors are more visible in maps, which have rigid geometry, than in aerial photographs, which are more chaotic.

We trained colorization on ImageNet [\[48\]](#page-9-2), and tested on the test split introduced by [\[58,](#page-9-0) [32\]](#page-8-0). Our method, with L1+cGAN loss, fooled participants on 22.5% of trials (Table [4\)](#page-6-1). We also tested the results of [\[58\]](#page-9-0) and a variant of their method that used an L2 loss (see [\[58\]](#page-9-0) for details). The conditional GAN scored similarly to the L2 variant of [\[58\]](#page-9-0) (difference insignificant by bootstrap test), but fell short of [\[58\]](#page-9-0)'s full method, which fooled participants on 27.8% of trials in our experiment. We note that their method was specifically engineered to do well on colorization.

### 4.6. Semantic segmentation

Conditional GANs appear to be effective on problems where the output is highly detailed or photographic, as is common in image processing and graphics tasks. What about vision problems, like semantic segmentation, where the output is instead less complex than the input?

To begin to test this, we train a cGAN (with/without L1 loss) on cityscape photo→labels. Figure [11](#page-7-0) shows qualitative results, and quantitative classification accuracies are reported in Table [5.](#page-6-2) Interestingly, cGANs, trained *without* the L1 loss, are able to solve this problem at a reasonable degree of accuracy. To our knowledge, this is the first demonstration of GANs successfully generating "labels", which are nearly discrete, rather than "images", with their continuousvalued variation[2](#page-6-3) . Although cGANs achieve some success, they are far from the best available method for solving this problem: simply using L1 regression gets better scores than using a cGAN, as shown in Table [5.](#page-6-2) We argue that for vision problems, the goal (i.e. predicting output close to ground truth) may be less ambiguous than graphics tasks, and reconstruction losses like L1 are mostly sufficient.

#### 4.7. Community-driven Research

Since the initial release of the paper and our pix2pix codebase, the Twitter community, including computer vision and graphics practitioners as well as artists, have successfully applied our framework to a variety of novel imageto-image translation tasks, far beyond the scope of the original paper. Figure [10](#page-7-1) shows just a few examples from the #pix2pix hashtag, such as *Sketch* → *Portrait*, *"Do as*

<span id="page-6-3"></span><sup>2</sup>Note that the label maps we train on are not exactly discrete valued, as they are resized from the original maps using bilinear interpolation and saved as jpeg images, with some compression artifacts.

<span id="page-7-2"></span>![](p7__page_7_Figure_0.jpeg)

Figure 9: Results of our method on several tasks (data from [\[42\]](#page-9-3) and [\[17\]](#page-8-1)). Note that the sketch→photo results are generated by a model trained on automatic edge detections and tested on human-drawn sketches. Please see online materials for additional examples.

<span id="page-7-1"></span>![](p7__page_7_Figure_2.jpeg)

Figure 10: Example applications developed by online community based on our pix2pix codebase: *#edges2cats* [\[3\]](#page-8-2) by Christopher Hesse, *Sketch* → *Portrait* [\[7\]](#page-8-3) by Mario Kingemann, *"Do As I Do" pose transfer* [\[2\]](#page-8-4) by Brannon Dorsey, *Depth*→ *Streetview* [\[5\]](#page-8-5) by Jasper van Loenen, *Background removal* [\[6\]](#page-8-6) by Kaihu Chen, *Palette generation* [\[4\]](#page-8-7) by Jack Qiao, and *Sketch*→ *Pokemon* [\[1\]](#page-8-8) by Bertrand Gondouin.

<span id="page-7-0"></span>![](p7__page_7_Figure_4.jpeg)

Figure 11: Applying a conditional GAN to semantic segmentation. The cGAN produces sharp images that look at glance like the ground truth, but in fact include many small, hallucinated objects.

*I Do" pose transfer*, *Depth*→*Streetview*, *Background removal*, *Palette generation*, *Sketch*→*Pokemon*, as well as the bizarrely popular #edges2cats.

## 5. Conclusion

The results in this paper suggest that conditional adversarial networks are a promising approach for many imageto-image translation tasks, especially those involving highly structured graphical outputs. These networks learn a loss adapted to the task and data at hand, which makes them applicable in a wide variety of settings.

Acknowledgments: We thank Richard Zhang, Deepak Pathak, and Shubham Tulsiani for helpful discussions, Saining Xie for help with the HED edge detector, and the online community for exploring many applications and suggesting improvements. This work was supported in part by NSF SMA-1514512, NGA NURI, IARPA via Air Force Research Laboratory, Intel Corp, Berkeley Deep Drive, and hardware donations by Nvidia.

# References

- <span id="page-8-8"></span>[1] Bertrand gondouin. [https://twitter.com/](https://twitter.com/bgondouin/status/818571935529377792) [bgondouin/status/818571935529377792](https://twitter.com/bgondouin/status/818571935529377792). Accessed, 2017-04-21. [8](#page-7-2)
- <span id="page-8-4"></span>[2] Brannon dorsey. [https://twitter.com/](https://twitter.com/brannondorsey/status/806283494041223168) [brannondorsey/status/806283494041223168](https://twitter.com/brannondorsey/status/806283494041223168). Accessed, 2017-04-21. [8](#page-7-2)
- <span id="page-8-2"></span>[3] Christopher hesse. [https://affinelayer.com/](https://affinelayer.com/pixsrv/) [pixsrv/](https://affinelayer.com/pixsrv/). Accessed: 2017-04-21. [8](#page-7-2)
- <span id="page-8-7"></span>[4] Jack qiao. <http://colormind.io/blog/>. Accessed: 2017-04-21. [8](#page-7-2)
- <span id="page-8-5"></span>[5] Jasper van loenen. [https://jaspervanloenen.](https://jaspervanloenen.com/neural-city/) [com/neural-city/](https://jaspervanloenen.com/neural-city/). Accessed, 2017-04-21. [8](#page-7-2)
- <span id="page-8-6"></span>[6] Kaihu chen. [http://www.terraai.org/](http://www.terraai.org/imageops/index.html) [imageops/index.html](http://www.terraai.org/imageops/index.html). Accessed, 2017-04-21. [8](#page-7-2)
- <span id="page-8-3"></span>[7] Mario klingemann. [https://twitter.com/](https://twitter.com/quasimondo/status/826065030944870400) [quasimondo/status/826065030944870400](https://twitter.com/quasimondo/status/826065030944870400). Accessed, 2017-04-21. [8](#page-7-2)
- [8] A. Buades, B. Coll, and J.-M. Morel. A non-local algorithm for image denoising. In *CVPR*, volume 2, pages 60–65. IEEE, 2005. [1](#page-0-0)
- [9] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. In *ICLR*, 2015. [2](#page-1-0)
- [10] T. Chen, M.-M. Cheng, P. Tan, A. Shamir, and S.-M. Hu. Sketch2photo: internet image montage. *ACM Transactions on Graphics (TOG)*, 28(5):124, 2009. [1](#page-0-0)
- [11] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. The cityscapes dataset for semantic urban scene understanding. In *CVPR)*, 2016. [4](#page-3-0)
- [12] E. L. Denton, S. Chintala, R. Fergus, et al. Deep generative image models using alaplacian pyramid of adversarial networks. In *NIPS*, pages 1486–1494, 2015. [2](#page-1-0)
- [13] A. Dosovitskiy and T. Brox. Generating images with perceptual similarity metrics based on deep networks. *arXiv preprint arXiv:1602.02644*, 2016. [2](#page-1-0)
- [14] A. A. Efros and W. T. Freeman. Image quilting for texture synthesis and transfer. In *SIGGRAPH*, pages 341–346. ACM, 2001. [1,](#page-0-0) [4](#page-3-0)
- [15] A. A. Efros and T. K. Leung. Texture synthesis by nonparametric sampling. In *ICCV*, volume 2, pages 1033–1038. IEEE, 1999. [4](#page-3-0)
- [16] D. Eigen and R. Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 2650–2658, 2015. [1](#page-0-0)
- <span id="page-8-1"></span>[17] M. Eitz, J. Hays, and M. Alexa. How do humans sketch objects? *SIGGRAPH*, 31(4):44–1, 2012. [4,](#page-3-0) [8](#page-7-2)
- [18] R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, and W. T. Freeman. Removing camera shake from a single photograph. In *ACM Transactions on Graphics (TOG)*, volume 25, pages 787–794. ACM, 2006. [1](#page-0-0)
- [19] L. A. Gatys, A. S. Ecker, and M. Bethge. Texture synthesis and the controlled generation of natural stimuli using convolutional neural networks. *arXiv preprint arXiv:1505.07376*, 12, 2015. [4](#page-3-0)

- [20] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. *CVPR*, 2016. [4](#page-3-0)
- [21] J. Gauthier. Conditional generative adversarial nets for convolutional face generation. *Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester*, 2014(5):2, 2014. [2](#page-1-0)
- [22] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In *NIPS*, 2014. [2,](#page-1-0) [4,](#page-3-0) [5,](#page-4-1) [6](#page-5-1)
- [23] A. Hertzmann, C. E. Jacobs, N. Oliver, B. Curless, and D. H. Salesin. Image analogies. In *SIGGRAPH*, pages 327–340. ACM, 2001. [1,](#page-0-0) [4](#page-3-0)
- [24] G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality of data with neural networks. *Science*, 313(5786):504–507, 2006. [3](#page-2-0)
- [25] S. Iizuka, E. Simo-Serra, and H. Ishikawa. Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification. *ACM Transactions on Graphics (TOG)*, 35(4), 2016. [2](#page-1-0)
- [26] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 2015. [3,](#page-2-0) [4](#page-3-0)
- [27] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. 2016. [2,](#page-1-0) [3](#page-2-0)
- [28] L. Karacan, Z. Akata, A. Erdem, and E. Erdem. Learning to generate images of outdoor scenes from attributes and semantic layouts. *arXiv preprint arXiv:1612.00215*, 2016. [2](#page-1-0)
- [29] D. Kingma and J. Ba. Adam: A method for stochastic optimization. *ICLR*, 2015. [4](#page-3-0)
- [30] P.-Y. Laffont, Z. Ren, X. Tao, C. Qian, and J. Hays. Transient attributes for high-level understanding and editing of outdoor scenes. *ACM Transactions on Graphics (TOG)*, 33(4):149, 2014. [1,](#page-0-0) [4](#page-3-0)
- [31] A. B. L. Larsen, S. K. Sønderby, and O. Winther. Autoencoding beyond pixels using a learned similarity metric. *arXiv preprint arXiv:1512.09300*, 2015. [3](#page-2-0)
- <span id="page-8-0"></span>[32] G. Larsson, M. Maire, and G. Shakhnarovich. Learning representations for automatic colorization. *ECCV*, 2016. [2,](#page-1-0) [7](#page-6-4)
- [33] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, ´ A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. *arXiv preprint arXiv:1609.04802*, 2016. [2](#page-1-0)
- [34] C. Li and M. Wand. Combining markov random fields and convolutional neural networks for image synthesis. *CVPR*, 2016. [2,](#page-1-0) [4](#page-3-0)
- [35] C. Li and M. Wand. Precomputed real-time texture synthesis with markovian generative adversarial networks. *ECCV*, 2016. [2,](#page-1-0) [4](#page-3-0)
- [36] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In *CVPR*, pages 3431– 3440, 2015. [1,](#page-0-0) [2,](#page-1-0) [4](#page-3-0)
- [37] M. Mathieu, C. Couprie, and Y. LeCun. Deep multi-scale video prediction beyond mean square error. *ICLR*, 2016. [2,](#page-1-0) [3](#page-2-0)
- [38] M. Mirza and S. Osindero. Conditional generative adversarial nets. *arXiv preprint arXiv:1411.1784*, 2014. [2](#page-1-0)

- [39] A. Owens, P. Isola, J. McDermott, A. Torralba, E. H. Adelson, and W. T. Freeman. Visually indicated sounds. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 2405–2413, 2016. [4](#page-3-0)
- [40] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros. Context encoders: Feature learning by inpainting. *CVPR*, 2016. [2,](#page-1-0) [3](#page-2-0)
- [41] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. *arXiv preprint arXiv:1511.06434*, 2015. [2,](#page-1-0) [3](#page-2-0)
- [42] R. S. Radim Tyle ˇ cek. Spatial pattern templates for recogni- ˇ tion of objects with regular structure. In *Proc. GCPR*, Saarbrucken, Germany, 2013. [4,](#page-3-0) [8](#page-7-0)
- [43] S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. *arXiv preprint arXiv:1605.05396*, 2016. [2](#page-1-0)
- [44] S. Reed, A. van den Oord, N. Kalchbrenner, V. Bapst, M. Botvinick, and N. de Freitas. Generating interpretable images with controllable structure. Technical report, Technical report, 2016. 2, 2016. [2](#page-1-0)
- [45] S. E. Reed, Z. Akata, S. Mohan, S. Tenka, B. Schiele, and H. Lee. Learning what and where to draw. In *Advances In Neural Information Processing Systems*, pages 217–225, 2016. [2](#page-1-0)
- [46] E. Reinhard, M. Ashikhmin, B. Gooch, and P. Shirley. Color transfer between images. *IEEE Computer Graphics and Applications*, 21:34–41, 2001. [6](#page-5-0)
- [47] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In *MIC-CAI*, pages 234–241. Springer, 2015. [2,](#page-1-0) [3](#page-2-0)
- [48] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. *IJCV*, 115(3):211–252, 2015. [4,](#page-3-0) [7](#page-6-0)
- [49] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. *arXiv preprint arXiv:1606.03498*, 2016. [2,](#page-1-0) [4](#page-3-0)
- [50] Y. Shih, S. Paris, F. Durand, and W. T. Freeman. Data-driven hallucination of different times of day from a single outdoor photo. *ACM Transactions on Graphics (TOG)*, 32(6):200, 2013. [1](#page-0-0)
- [51] D. Ulyanov, A. Vedaldi, and V. Lempitsky. Instance normalization: The missing ingredient for fast stylization. *arXiv preprint arXiv:1607.08022*, 2016. [4](#page-3-0)
- [52] X. Wang and A. Gupta. Generative image modeling using style and structure adversarial networks. *ECCV*, 2016. [2,](#page-1-0) [3,](#page-2-0) [4](#page-3-0)
- [53] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4):600–612, 2004. [2](#page-1-0)
- [54] S. Xie, X. Huang, and Z. Tu. Top-down learning for structured labeling with convolutional pseudoprior. 2015. [2](#page-1-0)
- [55] S. Xie and Z. Tu. Holistically-nested edge detection. In *ICCV*, 2015. [1,](#page-0-0) [2,](#page-1-0) [4](#page-3-0)
- [56] D. Yoo, N. Kim, S. Park, A. S. Paek, and I. S. Kweon. Pixellevel domain transfer. *ECCV*, 2016. [2,](#page-1-0) [3](#page-2-0)

- [57] A. Yu and K. Grauman. Fine-Grained Visual Comparisons with Local Learning. In *CVPR*, 2014. [4](#page-3-0)
- [58] R. Zhang, P. Isola, and A. A. Efros. Colorful image colorization. *ECCV*, 2016. [1,](#page-0-0) [2,](#page-1-0) [4,](#page-3-0) [7](#page-6-0)
- [59] J. Zhao, M. Mathieu, and Y. LeCun. Energy-based generative adversarial network. *arXiv preprint arXiv:1609.03126*, 2016. [2](#page-1-0)
- [60] Y. Zhou and T. L. Berg. Learning temporal transformations from time-lapse videos. In *ECCV*, 2016. [2,](#page-1-0) [3,](#page-2-0) [7](#page-6-0)
- [61] J.-Y. Zhu, P. Krahenb ¨ uhl, E. Shechtman, and A. A. Efros. ¨ Generative visual manipulation on the natural image manifold. In *ECCV*, 2016. [2,](#page-1-0) [4](#page-3-0)