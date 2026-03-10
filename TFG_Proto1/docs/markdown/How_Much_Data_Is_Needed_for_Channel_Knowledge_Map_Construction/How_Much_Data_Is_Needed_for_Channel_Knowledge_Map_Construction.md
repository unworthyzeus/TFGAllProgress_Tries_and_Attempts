# How Much Data Is Needed for Channel Knowledge Map Construction?

Xiaoli Xu<sup>®</sup>, Member, IEEE, and Yong Zeng<sup>®</sup>, Senior Member, IEEE

Abstract—Channel knowledge map (CKM) has been recently proposed to enable environment-aware communications by utilizing historical or simulation generated wireless channel data. This paper studies the construction of one particular type of CKM, namely channel gain map (CGM), by using a finite number of measurements or simulation-generated data, with model-based spatial channel prediction. We try to answer the following question: How much data is sufficient for CKM construction? To this end, we first derive the average mean square error (AMSE) of the channel gain prediction as a function of the sample density of data collection in offline CGM construction, as well as the number of data points used in online spatial channel gain prediction. To model the spatial variation of the wireless environment within each cell, we divide the CGM into subregions and estimate the channel parameters from the local data within each subregion. The parameter estimation error and the channel prediction error based on estimated channel parameters are derived as functions of the number of data points within the subregion. The analytical results may guide the CGM construction and utilization by determining the required spatial sample density for offline data collection and the number of data points to be used for online channel prediction, so that the desired level of channel prediction accuracy is guaranteed.

Index Terms— Channel gain map (CGM), environment-aware communication, spatial channel prediction, parameter estimation, average mean square error.

### I. INTRODUCTION

WITH the ever-increasing node density and channel dimension in wireless communication networks, the acquisition of real-time channel state information (CSI) purely relying on the conventional pilot based channel estimation becomes costly, and even infeasible in rate-hungry and delay-stringent applications. On the other hand, the abundant location-specific channel data and powerful data-mining capability of wireless networks make it possible to shift

Manuscript received 13 November 2023; revised 7 March 2024; accepted 28 April 2024. Date of publication 14 May 2024; date of current version 11 October 2024. This work was supported by the National Natural Science Foundation of China under Grant 62101118 and Grant 62071114. The associate editor coordinating the review of this article and approving it for publication was W. Ni. (Corresponding author: Yong Zeng.)

Xiaoli Xu is with the National Mobile Communications Research Laboratory, School of Information Science and Engineering, Southeast University, Nanjing 210096, China (e-mail: xiaolixu@seu.edu.cn).

Yong Zeng is with the National Mobile Communications Research Laboratory, School of Information Science and Engineering, Southeast University, Nanjing 210096, China, and also with the Purple Mountain Laboratories, Nanjing 211111, China (e-mail: yong\_zeng@seu.edu.cn).

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TWC.2024.3397964.

Digital Object Identifier 10.1109/TWC.2024.3397964

<span id="page-0-0"></span>![](p1__page_0_Figure_13.jpeg)

Fig. 1. An illustration of CKM concept.

from the conventional environment-unaware communication to future environment-aware communication, which facilitates CSI acquisition [1]. One promising technique for realizing environment-aware communication is by leveraging channel knowledge map (CKM), which makes use of the available channel knowledge learnt from the physical environment and/or the wireless measurements [2]. As shown in Fig. 1, CKM can be constructed by using location-tagged data from the physical-map assisted simulation or the actual channel measurements by communication devices. The site-specific CKM can be stored and managed at the base station (BS) and high-level CKM with lighter information could be maintained for inter-cell coordination. During the CKM utilization stage, the user's physical or virtual location is used to query the CKM for obtaining a priori channel knowledge between the BS and the user. With such a priori channel knowledge, the required online training overhead for real-time CSI acquisition can be greatly reduced [3]. In addition, the training results could be feedback to the construction algorithm for updating the CKM, so as to reflect the environment dynamics.

This paper focuses on the construction of a specific type of CKM, termed channel gain map (CGM), which provides channel gain knowledge at arbitrary location within the cell. The most straightforward approach for channel gain prediction is by utilizing stochastic channel models, such as the distance-dependent path loss model. However, such models only use the very high-level environment attributes, such as the environment type: urban, suburban or rural [4]. As a result, the channel gain is only coarsely related to the user location via its distance from the BS, without considering the specific environment. The coarse channel information is mainly used for performance comparison, but not sufficient for system optimization, especially when the future network has stringent performance requirements. On the other hand, CGM is a site-specific database constructed based on the actual local wireless environment that is learnt from the physical map

1536-1276 © 2024 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

or the actual channel measurements. The spatial and temporal correlation of channel gains make it possible to construct CGM by using a finite number of data samples.

Channel gains of wireless links are typically modelled by three components, i.e., the path loss, large-scale shadowing and small-scale fading. The path loss and shadowing are mainly affected by the propagation environment, e.g., the blockage due to building, terrain and plants, which can be considered as relatively static. On the other hand, the small-scale fading de-correlates fast in both the temporal and spatial domain, which is hard to be predicted in complex environment. Since the predictable part of channel gain, i.e., the path loss and shadowing, is relatively stable to the environment dynamics, this paper focuses on the *spatial* prediction for CGM construction. We aim to determine the required spatial sample density for achieving the desired level of accuracy in channel prediction.[1](#page-1-0) Intuitively, the larger the spatial sample density is, the more likely to find the data points that are close and hence highly correlated with the channel at the target location. However, larger sample density implies high construction and maintenance cost of the CGM. Besides, the effect of measurement error and small-scale multi-path fading can be mitigated by including more samples for online channel prediction. However, including more samples not only incurs larger computational cost, but also introduces bias if uncorrelated samples are counted.

The spatial prediction of channel gain can be achieved in different ways, e.g., the model-based prediction [\[5\],](#page-10-4) [\[6\],](#page-10-5) [\[7\], da](#page-10-6)ta-based prediction [\[8\],](#page-10-7) [\[9\],](#page-10-8) [\[10\]](#page-10-9) and the hybrid model and data-based prediction [\[11\]. F](#page-10-10)or model-based prediction, the large-scale path loss at the target location is modeled as a deterministic function of the distance between the target location and the BS. Then, the shadowing at the target location is estimated from the shadowing experienced by the neighboring locations in CGM, which is obtained by subtracting their respective path loss from the measured channel gain. Given the path loss parameters and the spatial correlation model [\[12\], t](#page-10-11)he minimum mean square error (MMSE) prediction has been proposed in [\[5\]. T](#page-10-4)he authors in [\[5\]](#page-10-4) also investigate the maximum likelihood (ML) and least square (LS) estimation of the channel parameters from the sample measurements. The extension of the spatial prediction algorithms to the spatial-temporal prediction are discussed in [\[13\]](#page-10-12) and [\[14\],](#page-10-13) where the temporal predictions are achieved by nonlinear filtering and autoregressive prediction, respectively. Since the path loss and spatial correlation between channel gains are modelled as functions of distances, the accuracy of the model-based spatial prediction is sensitive to localization error [\[15\].](#page-10-14) For data-based prediction, the channel gain is viewed as spatially distributed data and the geostatical tools [\[16\]](#page-10-15) are adopted to perform the spatial interpolation, such as the k-nearest neighbors (KNN), inverse-distance weight (IDW) and the Kriging Algorithms [\[8\]. Be](#page-10-7)sides, the collected data points can also be used to train a deep neural network

<span id="page-1-0"></span><sup>1</sup>The temporal prediction error may be modeled as a term in additional of the spatial prediction error. Besides, the temporal prediction error could be reduced by frequently updating the CGM to reflect environment change. Hence, only spatial prediction error is considered in this paper.

(DNN), which is then used to predict the channel gain at the target location [\[17\]. T](#page-10-16)he performance of the data-based prediction can be improved if the environment information is properly used, e.g., the 3D physical map is used to assist the DNN training process in [\[18\]. T](#page-10-17)he authors in [\[11\]](#page-10-10) considers the hybrid model and data based channel prediction, where the collected channel data is first divided into different groups based on the expectation maximization (EM) algorithm, and then the model-based spatial prediction is applied within each data group. However, such data division is only effective when the environment is simple so that the boundary of each group can be clearly identified.

In general, the model-based prediction algorithms exploit the expert knowledge and enable the estimation under limited data set. The data-driven methods focus on the spatial interpolation based on the measurement samples. Compared to model-based prediction, data-based methods are usually more flexible and may achieve better performance in complex environment if sufficient data is available. However, data-driven methods are not as tractable as the model-based methods. To get some insights on the question: "How much data is needed for CKM construction?", we treat the channel gain knowledge as location-tagged spatial data, and adopt the spatial correlation model used in the existing model-based prediction algorithms. The channel prediction performance is derived as a function of the spatial sample density in offline CGM construction, as well as the number of data points used in online channel prediction. Specifically, we consider two types of spatial sample distributions, i.e., the random distribution modelled by homogeneous Poisson point process (PPP) and the grid distribution with certain separation between adjacent samples. When the channel modeling parameters are known, we first derive the average mean square error (AMSE) of the channel gain prediction as a function of the spatial sample density, when only the nearest data point is used for online channel prediction. Then, the results are extended to a finite number of data points for online channel prediction. The analytical results show that the marginal improvement for AMSE decreases when more samples are included. Since the analysis is based on the spatial correlation of channel data, the derived relationship between the CGM density and the channel prediction performance is valid as long as the spatial correlation model is correct.

One of the main challenges for obtaining a correct channel model is that the wireless environment may vary significantly even within each site, e.g., the users may experience frequent shift between line-of-sight (LoS) link and non-LoS (NLoS) link as they move in urban environment. It was shown in [\[19\]](#page-10-18) that different channel gain models need to be used for users in different streets. Hence, it is desired to establish the local channel gain models for CGM construction. Given the sample density, the expected number of samples within the local region reduces with the region size. The limited number of samples may lead to large error in channel parameter estimation. Since the channel prediction is more sensitive to the path loss parameters, we derive the estimation error of the path loss exponent and path loss intercept as functions of the number of samples within the region. We also point out that the

<span id="page-2-2"></span>![](p1__page_2_Figure_2.jpeg)

Fig. 2. Distribution of data collecting locations. The red star is the location of the BS.

estimation error in channel parameters may not lead to error in channel prediction, since the correlated shadowing may be counted in the path loss intercept in a small region. Finally, the analytical results are verified by extensive simulations and a case study with actual CGM constructed from sparse samples is presented to validate the conclusions drawn from the analytical results.

The rest of this paper is organized as follows. The CGM construction problem is formulated in Section [II.](#page-2-0) Section [III](#page-2-1) presents the CGM-based spatial channel prediction methods and analyzes the achievable AMSE when the model parameters are known. Section [IV](#page-4-0) considers the channel parameter estimation and spatial prediction based on local measurements within a small region. The analytical expressions are verified by the numerical results in Section [V,](#page-6-0) and finally Section [VI](#page-8-0) draws the conclusion of this paper and summarizes the important lessons learnt from this study.

## II. SYSTEM MODEL

<span id="page-2-0"></span>We consider the BS of a macro cell that wants to construct a CGM to enable environment-aware communications. The CGM stores the channel gains at some limited sample locations, which are used to infer the channel gains at any location within the cell, based on the spatial predictability of the channel. According to [\[2\], th](#page-10-1)e CGM can be constructed either by channel measurements returned by network users at random locations, or by dedicated channel measurement devices at designed locations. It is difficult to characterize the real distribution of channel measurements due to the non-homogeneous user distribution and the complex physical environment that affects the device deployment. To gain some insights on the required sample density for achieving the desired channel prediction performance, we consider the following two distributions of the data measurement locations:

- *Random Sampling*: Data is collected at random locations, modeled by homogeneous PPP with density λ.
- *Grid Sampling*: Data is collected at grid points separated with a certain distance d.

Fig. [2](#page-2-2) presents an example for the random and grid sampling of the data collecting locations. The CGM constructed by the above two methods are referred to as *random CGM* and *grid CGM*, and denoted by M<sup>r</sup> and Mg, respectively. The location of the BS is denoted as the origin and the cell is denoted by A, with side length D. The number of data points to construct the CGM is λD<sup>2</sup> and ( D d ) 2 in random and grid CGM, respectively. According to the spatial predictability of the channel, the channel gain at any arbitrary location q ∈ A, denoted by ΥdB(q), can be estimated from the collected data stored in the CGM. The prediction function is written as

$$\tilde{\Upsilon}_{dB}(\mathbf{q}) = f(\mathbf{q}, \mathcal{M}_c), c \in \{r, g\}.$$
 (1)

The AMSE of the spatial prediction over all target locations within the area is defined as

$$AMSE = \frac{1}{|\mathcal{A}|} \int_{\mathbf{q} \in \mathcal{A}} \left( \Upsilon_{dB}(\mathbf{q}) - \tilde{\Upsilon}_{dB}(\mathbf{q}) \right)^2 d\mathcal{A}, \quad (2)$$

where |A| is the size of the area.

Intuitively, the accuracy of the spatial channel prediction improves with the density of the data collecting locations, represented by λ and d in M<sup>r</sup> and Mg, respectively. Besides, the complexity of the estimation algorithm also plays an important role in the spatial prediction, which can be reflected by the number of neighboring data points in CGM used for online channel prediction. In order to answer the question "how much data is needed for CKM construction?", we derive the relationship between the AMSE with the data collection density and data used for online prediction.

# <span id="page-2-1"></span>III. CGM-BASED CHANNEL PREDICTION WITH KNOWN PARAMETERS

In the wireless communication literature, it is well established that the channel gain constitutes of three major components, i.e., the path loss, shadowing and multipath fading. Denote by Υ(q) the channel gain between the BS located at the origin and a receiver at location q. We have Υ(q) = ΥPL(q)ΥSH(q)ΥMP(q), where ΥPL(q), ΥSH(q) and ΥMP(q) represent the impact of path loss, shadowing and multipath fading, respectively. With ΥdB(q) = 10 log10(Υ(q)) being the channel gain in dB, we have

$$\Upsilon_{\mathrm{dB}}(\mathbf{q}) = K_{\mathrm{dB}} - 10n_{\mathrm{PL}}\log_{10}(\|\mathbf{q}\|) + v(\mathbf{q}) + \omega(\mathbf{q}),$$
 (3)

where KdB accounts the path loss intercept, nPL is the path loss exponent, v(q) = 10 log10(ΥSH(q)) is a random variable that models the effect of shadowing, and ω(q) = 10 log10(ΥMP(q)) models the effect of multipath fading. Generally, v(q) can be modelled as a zero-mean Gaussian random variable with variance α and spatial correlation function

<span id="page-2-3"></span>
$$\varepsilon(h) = \alpha \exp\left(-\frac{h}{\beta}\right),$$
 (4)

where h is the distance between the pair of data points and β is known as the correlation distance of shadowing [\[12\]. I](#page-10-11)n other words, we have E{v(qi)v(q<sup>j</sup> )} = ε(∥qi−qj∥). The multipath effect ω(q) is also modeled as a zero-mean Gaussian random variable, with variance σ 2 and it spatially de-correlates fast as compared with the shadowing.

## *A. Spatial Prediction of Channel Gain*

We first assume that the channel modeling parameters in [\(3\),](#page-2-3) i.e., {nPL, KdB, α, β, σ<sup>2</sup>}, are fixed and known. Then, the estimation of channel gain at location q is equivalent to estimating the shadowing effect v(q) since the path loss can be computed from the distance to the BS and the multipath fading can be hardly predicted, and hence treated as noise.

Ideally, all data points in the CGM should be used for estimating  $\Upsilon_{\rm dB}(\mathbf{q})$ . However, since the channel correlation diminishes with the separation distance and for the purpose of reducing the computational complexity, we assume that only k nearest data collecting locations are included for online channel prediction. Denote the locations of k nearest data point in CGM by  $\mathcal{Q}_k = \{\mathbf{q}_1, \dots, \mathbf{q}_k\}$ . According to [5], the MMSE estimation of  $v(\mathbf{q})$  is given by

$$\tilde{v}(\mathbf{q}) = \phi_O^T(\mathbf{q})(\mathbf{R}_Q + \sigma^2 \mathbf{I}_{k \times k})^{-1}(\mathbf{y}_Q - K_{dB} - n_{PL}\mathbf{h}),$$
 (5)

where  $\phi_Q(\mathbf{q}) = \left[\varepsilon(\|\mathbf{q} - \mathbf{q}_1\|), \cdots, \varepsilon(\|\mathbf{q} - \mathbf{q}_k\|)\right]^T$  captures the correlation between the shadowing at the target location  $\mathbf{q}$  and that at all the data collecting locations in  $\mathcal{Q}_k$ ,  $\mathbf{R}_Q$  is the correlation matrix, with the (i,j)th component being  $[\mathbf{R}_Q]_{ij} = \varepsilon(\|\mathbf{q}_i - \mathbf{q}_j\|)$ ,  $\mathbf{I}_{k\times k}$  is the identity matrix,  $\mathbf{y}_Q = [\Upsilon_{\mathrm{dB}}(\mathbf{q}_1), \cdots, \Upsilon_{\mathrm{dB}}(\mathbf{q}_k)]^T$  is the measurements stored in the CGM, and  $\mathbf{h} = [-10\log_{10}(\|\mathbf{q}_1\|), \cdots, -10\log_{10}(\|\mathbf{q}_k\|)]^T$  is the distance vector.

The estimated channel gain  $\tilde{\Upsilon}_{dB}(\mathbf{q})$  is the summation of the path loss and the estimated shadowing  $\tilde{v}(\mathbf{q})$ , i.e.,

$$\tilde{\Upsilon}_{dB}(\mathbf{q}) = K_{dB} - 10n_{PL}\log_{10}(\|\mathbf{q}\|) + \tilde{v}(\mathbf{q}). \tag{6}$$

The complexity of the online channel prediction according to (6) mainly attributes to the inversion of a  $k \times k$  matrix in (5), which can be reduced by considering less number of neighboring data points, i.e., using a smaller k. Besides, the MSE of the estimation in (6) is given by

$$\xi_{\mathrm{dB}}(\mathbf{q}) = \left(\Upsilon_{\mathrm{dB}}(\mathbf{q}) - \tilde{\Upsilon}_{\mathrm{dB}}(\mathbf{q})\right)^{2}$$
$$= \alpha + \sigma^{2} - \phi_{Q}^{T}(\mathbf{q})(\mathbf{R}_{Q} + \sigma^{2}\mathbf{I}_{k \times k})^{-1}\phi_{Q}(\mathbf{q}). \quad (7)$$

By substituting (7) into (2) and taking average with respect to the random location  $\mathbf{q} \in \mathcal{A}$ , we can get the AMSE for the CGM constructed by random and grid sampling, respectively, as elaborated below.

## B. AMSE of Random CGM

We start by considering the special case when only the nearest data point is used, i.e., k=1. Denote by  $\mathbf{q}_1$  the location of the nearest data point stored in CGM and let  $d_{\min} = \|\mathbf{q} - \mathbf{q}_1\|$ . Then, the correlation matrix is simplified to  $\mathbf{R}_Q = 1$  and the estimated shadowing loss in (5) reduces to

$$\tilde{v}(\mathbf{q}) = \frac{\alpha}{1 + \sigma^2} \exp\left(-\frac{d_{\min}}{\beta}\right) v(\mathbf{q}_1),$$
 (8)

where  $v(\mathbf{q}_1) = \Upsilon_{\mathrm{dB}}(\mathbf{q}_1) - K_{\mathrm{dB}} + 10n_{\mathrm{PL}}\log_{10}(\|\mathbf{q}_1\|)$  is the shadowing loss at  $\mathbf{q}_1$ . Furthermore, the MMSE in (7) is simplified to

$$\xi_{\mathrm{dB}}(\mathbf{q})|_{(k=1)} = \alpha + \sigma^2 - \frac{\alpha^2}{\alpha + \sigma^2} \exp\left(-\frac{2d_{\min}}{\beta}\right).$$
 (9)

When the data points are distributed randomly, according to the homogeneous PPP with density  $\lambda$ , the distribution of  $d_{\min}$  is given by the probability density function (p.d.f.)  $P_r(x)$  [20] with

$$P_r(x) = 2\pi\lambda x \exp\left(-\pi\lambda x^2\right). \tag{10}$$

By substituting (9) and (10) into (2), we can obtain the AMSE as a function of the density  $\lambda$ , which is presented in Lemma 1.

<span id="page-3-5"></span>Lemma 1: If the CGM is constructed by data collection at random locations according to the homogeneous PPP with density  $\lambda$  and only the nearest data point is used for online channel prediction, the AMSE is given by

$$AMSE = \mathbb{E}[\xi_{\mathrm{dB}}(\mathbf{q})|_{(k=1)}] = \alpha + \sigma^2 - \frac{\alpha^2}{\alpha + \sigma^2} \zeta_r(\lambda), (11)$$

<span id="page-3-1"></span>where

<span id="page-3-6"></span>
$$\begin{split} \zeta_r(\lambda) &= \int_0^\infty \exp\left(-\frac{2x}{\beta}\right) P_r(x) dx \\ &= 2\pi \lambda \int_0^\infty x \exp\left(-\pi \lambda x^2 - \frac{2x}{\beta}\right) dx \\ &= 1 - \frac{1}{\beta\sqrt{\lambda}} \exp\left(\frac{1}{\pi\lambda\beta^2}\right) \left(1 - \Phi\left(\frac{1}{\beta\sqrt{\pi\lambda}}\right)\right) \end{split}$$

and  $\Phi(x)=\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}dt$  is the Gauss error function. Proof: Please refer to Appendix A.

<span id="page-3-0"></span>When  $\lambda \to 0$ , the AMSE in (11) reduces to  $(\alpha + \sigma^2)$ , which corresponds to the variance of the channel gain prediction using path loss only. When  $\lambda \to \infty$ ,  $\zeta_r(\lambda) \to 1$  and (11) reduces to  $\left(\sigma^2 + \frac{\alpha\sigma^2}{\alpha + \sigma^2}\right)$ , representing the limit of channel gain spatial prediction. In general,  $\zeta_r(\lambda)$  is an increasing function with  $\lambda$  and hence the AMSE decreases with the spatial sample density  $\lambda$ . Besides, the spatial prediction is more effective, i.e., with a larger decaying slope with  $\lambda$ , when the shadowing variance is larger than the fading power.

<span id="page-3-2"></span>Next, we derive the AMSE for general value of k, which is rather challenging since it involves the inverse of a random matrix  $\mathbf{R}_Q$ . To gain some insights, we first consider the asymptotical behaviors. When the CGM density is sufficiently large, i.e.,  $\lambda \to \infty$ , or the spatial correlation vanishes, i.e.,  $\beta \to \infty$ , we have  $\phi_Q \to \alpha \mathbf{1}_{k \times 1}$  and  $R_Q \to \mathbf{1}_{k \times k}$ , where  $\mathbf{1}_{m \times n}$  is a matrix of size  $m \times n$  with all the components being 1. Then, the inverse of the matrix can be obtained as

<span id="page-3-7"></span>
$$(\mathbf{R}_Q + \sigma^2 \mathbf{I}_{k \times k})^{-1} = (\alpha \mathbf{1}_{k \times k} + \sigma^2 \mathbf{I}_{k \times k})^{-1}$$
$$= \frac{1}{\sigma^2} \mathbf{I}_{k \times k} + \frac{\alpha}{\sigma^2 (k\alpha + \sigma^2)} \mathbf{1}_{k \times k}. \quad (12)$$

By substituting (12) and  $\Phi_Q$  into (9), we have

$$\lim_{\lambda \to \infty} \xi_{dB}(\mathbf{q}) = \lim_{\beta \to \infty} \xi_{dB}(\mathbf{q}) = \alpha + \sigma^2 - \frac{k\alpha^2}{k\alpha + \sigma^2}.$$
 (13)

<span id="page-3-3"></span>Since the limiting behavior of  $\xi_{\rm dB}({\bf q})$  no longer depends on the target location, we can obtain AMSE directly. Specifically, for CGM with general density  $\lambda$  and prediction algorithm including k neighboring data points, we can obtain the approximate AMSE by assuming that all the k measurements have the minimum distance with the target location, i.e., setting  $\phi_Q({\bf q}) \approx \alpha \exp\left(-\frac{d_{\min}}{\beta}\right), \forall {\bf q},$  and assuming that the measurements are close to each other, i.e.,  ${\bf R}_Q \approx {\bf 1}_{k \times k}$ . The obtained AMSE is presented in Lemma 2.

<span id="page-3-8"></span><span id="page-3-4"></span>Lemma 2: If the CGM is constructed by data collection at random locations according to the homogeneous PPP with

*density* λ *and* k *nearest data points are used for online channel prediction, the achieved AMSE is approximated by*

$$AMSE \approx \alpha + \sigma^2 - \frac{k\alpha^2}{k\alpha + \sigma^2} \zeta_r(\lambda),$$
 (14)

*where the approximation is tight when* k = 1 *or* λ → ∞*.*

Similar to [\(11\),](#page-3-0) when λ → 0, the sample locations are too far to be correlated, and hence the AMSE is (α+σ 2 ), which is independent of k. When λ is sufficiently large, the shadowing can be completely known from k → ∞ samples, and hence the resultant AMSE only contains the fading power σ 2 . In general, for given spatial sample density, the AMSE decays with k with the slope

$$\frac{\partial AMSE}{\partial k} = -\frac{\alpha^2 \sigma^2}{(k\alpha + \sigma^2)^2} \zeta_r(\lambda), \tag{15}$$

whose absolute value decays with k and increases with λ. This implies that the marginal improvement of AMSE decreases when k is sufficiently large, and including more data points is effective only when the spatial sample density is large.

## *C. AMSE of Grid CGM*

When the data collecting locations in CGM are placed in the grid with separation d, the distance between a random location q with the nearest measurement is distributed according to Pg(x) with

$$P_g(x) = \begin{cases} \frac{2\pi x}{d^2}, & 0 \le x \le \frac{d}{2} \\ \frac{4x}{d^2} \left(\frac{\pi}{2} - 2\arccos\left(\frac{d}{2x}\right)\right), & \frac{d}{2} \le x \le \frac{\sqrt{2}d}{2} \end{cases}$$

$$\tag{16}$$

The derivation of [\(16\)](#page-4-0) is presented in Appendix [B.](#page-9-0) Then, following similar analysis as for random CGM, we can obtain the AMSE for grid CGM with k = 1, as presented in Lemma [3.](#page-4-1)

<span id="page-4-1"></span>*Lemma 3: If the CGM is constructed by data collected at grid points with separation* d *and only the nearest data point is used for online channel gain prediction, the achieved AMSE is given by*

$$AMSE = \mathbb{E}[\xi_{\text{dB}}(\mathbf{q})|_{(k=1)}] = \alpha + \sigma^2 - \frac{\alpha^2}{\alpha + \sigma^2} \zeta_g(d) \quad (17)$$

*where*

$$\zeta_g(d) = \int_0^{\frac{d}{2}} \frac{2\pi x}{d^2} e^{-\frac{2x}{\beta}} dx + \int_{\frac{d}{2}}^{\frac{\sqrt{2}d}{2}} \frac{8x}{d^2} e^{-\frac{2x}{\beta}} \arccos\left(\frac{d}{2x}\right) dx$$
$$= \left[\frac{\pi\beta^2}{2d^2} \left(1 - \left(1 + \frac{\sqrt{2}d}{\beta}\right) e^{-\sqrt{2}d/\beta}\right) - 2\Psi\left(\frac{d}{\beta}\right)\right]$$

and 
$$\Psi(x) = \int_{1}^{\sqrt{(2)}} u e^{-ux} \left(\arccos\left(\frac{1}{u}\right)\right) du \approx (0.2854 - 0.0725x + 0.0108x^2)e^{-x}.$$

Following similar analysis as that for random CGM, we can obtain the same asymptotical MSE in [\(13\)](#page-3-1) for grid CGM when d → 0. Furthermore, by using similar approximation, we have *Lemma 4: If the CGM is constructed by data collected at*

*grid points with separation* d *and* k *nearest data points are*

<span id="page-4-5"></span><span id="page-4-3"></span>![](p5__page_4_Figure_18.jpeg)

Fig. 3. An illustration of the channel model variation within a site.

*used for online channel gain prediction, the achieved AMSE is given by*

<span id="page-4-2"></span>
$$AMSE \approx \alpha + \sigma^2 - \frac{k\alpha^2}{k\alpha + \sigma^2} \zeta_g(d),$$
 (18)

*where the approximation is tight when* k = 1 *or* d → 0*.*

Comparing [\(18\)](#page-4-2) with [\(14\),](#page-4-3) it is observed that the channel prediction performance in random CGM and grid CGM have the same limiting performance at the extremely large or small spatial sample density, i.e., with limd→<sup>0</sup> ζg(d) = 1 and limd→∞ ζg(d) = 0. In general, we have ζg(d) > ζr(λ) when they have the same effective density, i.e., with λ = 1/d<sup>2</sup> .

The analytical results presented in Lemma [2](#page-3-2) and Lemma [4](#page-4-4) may be used to guide the CGM construction and utilization procedures, by determining the required sample density for offline CKM construction and the number of data points to be used for online channel prediction, so that the desired level of channel prediction accuracy is guaranteed.

## <span id="page-4-0"></span>IV. CGM-BASED CHANNEL PREDICTION WITH UNKNOWN PARAMETERS AND SPATIAL VARIATIONS

In practice, the channel modeling parameters in the path loss model [\(3\)](#page-2-0) are usually unknown and they vary across different regions [\[19\].](#page-10-0) An illustration example is shown in Fig. [3,](#page-4-5) where Fig. [3\(a\)](#page-4-5) shows a city map and Fig. [3\(b\)](#page-4-5) shows the corresponding CGM, generated by the commercial ray tracing software Wireless Insite.[2](#page-4-6) It is clearly noted that the regions bounded by the three red boxes suffer from different levels of path loss, due to the different levels of blockage. Hence, different channel gain models should be used for channel spatial prediction. To this end, we divide the area into small regions and achieve the spatial prediction in two steps:

- Build the local channel gain model based on the data points within each small region, which is either determined by the distance threshold or building borders.
- For target location within each region, estimate the channel gain based on the local channel model and the data points in proximity of the target location.

There are two approaches for region division in the literature, i.e., the physical-map assisted approach [\[19\]](#page-10-0) and the data-driven approach [\[11\]. I](#page-10-1)n the former approach, the whole map is divided into different regions based on the building distribution and street orientation. In the latter approach, the

<span id="page-4-6"></span><span id="page-4-4"></span><sup>2</sup>https://www.remcom. com/wireless-insite-em-propagation-software

expectation maximization algorithm is used to classify the data into different groups, and each group corresponds a region. Better region division may be achieved by utilizing both the physical map information and channel measurements. In the following, we assume that the region division has been completed and consider the channel modeling parameter estimation for a local region with N measurements in the CGM, located at  $\{\mathbf{q}_1,\ldots,\mathbf{q}_N\}$ . Furthermore, we analyze the impact of spatial sample density and region size on the channel parameter estimation and the channel prediction.

## A. Estimation of Path Loss Parameters

Due to the correlation of shadowing, the joint maximum likelihood (ML) estimation of all the channel parameters is challenging and not tractable. To investigate the impact of CGM density on channel parameter estimations, we adopt the least square (LS) estimation, which is close-to-optimal if the shadowing correlation distance is small. Since the effects of shadowing and multipath fading are modeled as the zero-mean random variables, we can obtain the LS estimation of the path loss intercept and path loss exponent as

$$\left[\hat{K}_{\mathrm{dB}} \ \hat{n}_{\mathrm{PL}}\right] = \arg\min \|\mathbf{y} - K_{\mathrm{dB}} - n_{\mathrm{PL}}\mathbf{h}\|^2, \tag{19}$$

where  $\mathbf{y}$  is a vector containing the collected channel gain at all the N sample locations within the region,  $\mathbf{h} = [-10\log_{10}(\|\mathbf{q}_1\|), -10\log_{10}(\|\mathbf{q}_2\|), \dots -10\log_{10}(\|\mathbf{q}_N\|)]$  is the vector containing the distance of the N data points to the BS. Let  $\mathbf{H} = [\mathbf{1}_{N\times 1}, \mathbf{h}]$ , and the LS estimation of the path loss intercept and path loss exponents is given by

$$\begin{bmatrix} \hat{K}_{\text{dB}} \\ \hat{n}_{\text{PL}} \end{bmatrix} = (\mathbf{H}^T \mathbf{H})^{-1} \mathbf{H}^T \mathbf{y}. \tag{20}$$

The covariance matrix of the estimation error vector is

$$\mathbf{C}_{LS} = (\mathbf{H}^T \mathbf{H})^{-1} \mathbf{H}^T \left( \mathbf{R}_Q + \sigma^2 \mathbf{I}_{N \times N} \right) \mathbf{H} (\mathbf{H}^T \mathbf{H})^{-1}, \quad (21)$$

where  $\mathbf{R}_Q$  is the shadowing correlation matrix for N data points. The estimation error for path loss intercept  $K_{\mathrm{dB}}$  and path loss exponent  $n_{\mathrm{PL}}$  are given by

$$\sigma_{\hat{K}_{\text{dB}}}^2 = \mathbf{C}_{LS}(1,1), \sigma_{\hat{n}_{\text{PL}}}^2 = \mathbf{C}_{LS}(2,2).$$
 (22)

In general, the estimation errors in (22) depend on the number of data points and their locations. To gain some insights on the impact of region size and sample density, we assume that the distance of the data collection location to the BS is a random number uniformly distributed within the range  $[\delta_{\min}, \delta_{\max}]$ , i.e.,  $\|\mathbf{q}\| \sim \mathcal{U}(\delta_{\min}, \delta_{\max})$ . Then, we can approximate the channel parameter estimation error as a function of the number of data points N, and their distribution parameters  $(\delta_{\min}, \delta_{\max})$ , as stated in Lemma 5.

<span id="page-5-1"></span>Lemma 5: Consider a typical region containing N data points in the CGM with random point density  $\lambda$  or grid point separation d. If the distances of the measurement locations to the BS are uniformly distributed within  $[\delta_{\min}, \delta_{\max}]$ , the LS estimation error of the path loss exponent and path loss intercept in (21) can be approximated by

$$\sigma_{\hat{K}_{\text{dB}}}^2 = \frac{\alpha + \sigma^2/c}{N/c} \cdot \frac{\mu^2 + \chi}{\gamma} \tag{23}$$

$$\sigma_{\hat{n}_{\rm PL}}^2 = \frac{\alpha + \sigma^2/c}{N/c} \cdot \frac{1}{\chi} \tag{24}$$

whore

$$\begin{split} c &= \begin{cases} \max\{1, \pi\lambda\beta^2\}, & \text{for random CGM} \\ \max\{1, \pi\beta^2/d^2\}, & \text{for grid CGM} \end{cases} \\ \mu &= \frac{10\delta_{\max}\log_{10}(\delta_{\max}) - 10\delta_{\min}\log_{10}(\delta_{\min})}{\delta_{\max} - \delta_{\min}} - \frac{10}{\ln 10} \\ \chi &= \frac{100}{(\ln 10)^2} - \frac{100\delta_{\max}\delta_{\min}\log_{10}(\delta_{\max}/\delta_{\min})}{(\delta_{\max} - \delta_{\min})^2} \\ \textit{Proof: Please refer to Appendix C.} \end{split}$$

When the shadowing correlation distance  $\beta$  is small, we have c=1. Lemma 5 indicates that the estimation error is inversely proportional with the number of data points within the region, which implies that we can improve the parameter estimation accuracy by using a denser CGM. However, when  $\beta$  is large, increasing the CGM density will not change the value N/c and hence the improvement over the channel parameter estimation is rather limited. Besides, the estimation error also depends on the region size, measured by  $(\delta_{\max} - \delta_{\min})$ , within which the channel parameters are consistent.

## B. Estimation of Shadowing and Fading Parameters

The shadowing and fading parameters can be estimated using LS method or the weight LS fitting algorithms discussed in [21]. This paper adopts the LS estimation for its simplicity and tractability. Specifically, after obtaining the estimated path loss parameters in (20), the shadowing and fading at each data collecting location can be obtained by subtracting the path loss, i.e.,

$$\mathbf{s} = \mathbf{y} - \hat{K}_{\mathrm{dB}} - \hat{n}_{\mathrm{PL}} \mathbf{h},\tag{25}$$

<span id="page-5-3"></span><span id="page-5-2"></span>where  $\hat{K}_{\text{dB}}$  and  $\hat{n}_{\text{PL}}$  are given in (20).

<span id="page-5-0"></span>Consider a pair of data points at the locations  $\mathbf{q}_i$  and  $\mathbf{q}_j$ , with residual channel gain  $\mathbf{s}(i)$  and  $\mathbf{s}(j)$ , respectively, after subtracting the estimated path loss. The correlation between these residual channel gains is calculated as  $\mathbf{s}(i)\mathbf{s}(j)$ , which can be used to estimate the shadowing variance and correlation distance in (4). Specifically, let  $\mathcal{D}_0 = \{d_1, d_2, \ldots\}$  denote the set of all possible distances between a pair sample locations. To ensure that there are sufficient data to obtain the correlation value related with each distance, we group the similar distance in  $\mathcal{D}_0$  together. For example, if  $|d_1 - d_2| \leq \varepsilon_d$ , where  $\varepsilon_d$  is a small number, they are replaced by a single distance  $d = \frac{1}{2}(d_1 + d_2)$ . The resultant distance set is denoted by  $\mathcal{D}$ , with cardinality much smaller than  $\mathcal{D}_0$ .

Denote by  $\mathcal{I}_u = \{(i,j) : |(\|\mathbf{q}_i - \mathbf{q}_j\| - d_u)| \le \varepsilon_d\}$  the set of pairs with distance  $d_u$  for  $u = 1, \ldots, |\mathcal{D}|$ . Then, we can estimate the value of the correlation function at  $d_u$  as

$$\hat{\varepsilon}(d_u) = \frac{1}{|\mathcal{I}_u|} \sum_{(i,j)\in\mathcal{I}_u} \mathbf{s}(i)\mathbf{s}(j). \tag{26}$$

To reduce the complexity and ensure the meaningful estimation results, the cardinality of distance vector  $|\mathcal{D}|$  is further reduced by discarding the distance beyond which negative experimental correlation is observed. Mathematically, after

this procedure, we have  $\hat{\varepsilon}(d_u) > 0$  for  $u \leq |\mathcal{D}|$ . Then, the estimation of shadowing parameters can be formulated as

$$[\hat{\alpha}, \hat{\beta}] = \arg\min \sum_{d_u \in \mathcal{D}} |\mathcal{I}_u| \left( \ln(\alpha e^{-d_u/\beta}) - \ln(\hat{\varepsilon}(d_u)) \right),$$
(27)

where  $|\mathcal{I}_u|$  is set as the weight measuring the reliability of the experimental value  $\hat{\varepsilon}(d_u)$ . Problem (27) can be solved easily with the solution given by [5]

$$\begin{bmatrix} \ln(\hat{\alpha}) \\ \frac{1}{\beta} \end{bmatrix} = (\mathbf{M}^T \mathbf{W} \mathbf{M})^{-1} \mathbf{M}^T \mathbf{W} \begin{bmatrix} \ln(\hat{\varepsilon}(d_1)) \\ \vdots \\ \ln(\hat{\varepsilon}(d_{|\mathcal{D}|})) \end{bmatrix}, \quad (28)$$

where

$$\mathbf{M} = \begin{bmatrix} 1, & \cdots & 1 \\ -d_1 & \cdots & -d_{|\mathcal{D}|} \end{bmatrix}^T \text{ and } \mathbf{W} = \operatorname{diag}(|\mathcal{I}_1|, \cdots, |\mathcal{I}_{\mathcal{D}}|).$$

With  $\hat{\alpha}$  obtained in (28), the estimated multipath fading variance is given by

$$\hat{\sigma}^2 = \max\{\mathbf{s}^T \mathbf{s} - \hat{\alpha}, 0\}. \tag{29}$$

## C. AMSE of Spatial Channel Prediction

Based on the estimated channel modeling parameters within the local region, we can perform the spatial prediction of the channel gain within the region as in (6), by substituting the corresponding estimated channel parameters  $\{\hat{n}_{\rm PL}, \hat{K}_{\rm dB}, \hat{\alpha}, \hat{\beta}, \hat{\sigma}^2\}$ . It is observed that the AMSE of channel gain spatial prediction is not sensitive to the estimation error of the correlation distance and shadowing power, as compared with the path loss parameters. Hence, we focus on analyzing the impact of path loss estimation error on the AMSE.

Under the special case when the correlation distance  $\beta$  is smaller than the separation between samples, the channel gain estimation reduces to the estimation of path loss, and hence the AMSE is proportional to the path loss parameter estimation error, given in Lemma 5. Specifically, the channel gain estimation at q is given by

$$\tilde{\Upsilon}_{dB}(\mathbf{q}) = \hat{K}_{dB} - 10\hat{n}_{PL}\log_{10}(\|\mathbf{q}\|)$$
(30)

The MSE of the path loss estimation is given by

$$\xi_{\text{dB}}(\mathbf{q}) = \mathbb{E}[(\tilde{\Upsilon}_{\text{dB}} - \Upsilon_{\text{dB}})^{2}] 
= \alpha + \sigma^{2} + \sigma_{\hat{K}_{\text{dB}}}^{2} + (10\log_{10}(\|\mathbf{q}\|))^{2}\sigma_{\hat{n}_{\text{PL}}}^{2} 
- 10\log_{10}(\|\mathbf{q}\|))(\mathbf{C}_{\text{LS}}(2, 1) + \mathbf{C}_{\text{LS}}(1, 2))$$
(31)

where  $\sigma_{\hat{K}_{\mathrm{dB}}}^2$  and  $\sigma_{\hat{n}_{\mathrm{PL}}}^2$  are given by (23) and (24), respectively. Further assuming  $\|\mathbf{q}\| \sim \mathcal{U}(\delta_{\min}, \delta_{\max})$ , we get the AMSE as

$$AMSE = \mathbb{E}[\xi_{dB}(\mathbf{q})] = (\alpha + \sigma^2) \left(1 + \frac{2}{N}\right).$$
 (32)

which decreases monotonically with the number of data points  ${\cal N}$  within the region.

On the other hand, when  $\beta$  is large, the path loss at the target location is calculated from the estimated parameters and the shadowing is estimated by the shadowing experience by the neighboring measurements within the same region.

<span id="page-6-2"></span><span id="page-6-0"></span>![](p7__page_6_Figure_20.jpeg)

<span id="page-6-1"></span>Fig. 4. The AMSE of channel gain prediction with various CGM densities. For fair comparison, the separation distance in Grid CGM is set as  $d = 1/\sqrt{\lambda}$ .

In this case, the AMSE does not necessarily increase with the parameter estimation error, especially when the region is small. This is because the correlated shadowing loss can also be counted into the path loss intercept  $\hat{K}_{\rm dB}$ , without affecting the accuracy of channel gain prediction. Hence, deriving the AMSE is challenging. For most of the practical scenarios, we can approximate the AMSE by the case with known channel parameters, presented in Lemma 1 and Lemma 2.

## V. NUMERICAL RESULTS

The analysis presented in the preceding sections is verified by the numerical results in this section. We consider the CGM construction in a  $D \times D$  square area. Unless otherwise stated, the channel gain is generated according to the model in (3) with channel parameters  $n_{\rm PL}=2.2,~K_{\rm dB}=-80,~\alpha=8,~\beta=30$  m and  $\sigma^2=2$ .

First, we verify the analytical AMSE expressions presented in Lemma 1 and Lemma 3 for k=1 and various CGM densities in Fig. 4. It is observed that the analysis matches very well with the simulation results, even though an approximation is adopted to obtain the closed-form expression for  $\zeta_g(d)$  in Lemma 3. Besides, the AMSE monotonically decreases with the CGM density. The price paid for denser CGM is the higher construction and storage cost. With the same density, grid CGM slightly outperforms the random CGM since the distance between the random target locations and measurements are guaranteed to be no greater than  $\sqrt{2}d/2$ .

Under the fixed CGM density, the prediction accuracy can be improved by considering more neighboring data points in online channel prediction, i.e., with a larger k, as indicated by Lemma 2 and Lemma 4. The analytical results in (14) and (18) are verified in Fig. 5(a) and Fig. 5(b), respectively. As expected, the analytical results are tight when k=1 and rather accurate when CGM density is large. It is observed that the nearest data point provides the most information for channel prediction at the target location. With further increase of k, the performance improvement is limited, and it becomes almost flat far before k reaches the number of neighboring measurements within the correlation distance, i.e.,  $\pi\beta^2\lambda$  in random CGM and  $\left(\frac{\beta}{d}\right)^2$  in grid CGM. This justifies for the

![](p8__page_7_Figure_2.jpeg)

Fig. 5. The AMSE of channel gain prediction using various number of data points.

<span id="page-7-0"></span>![](p8__page_7_Figure_4.jpeg)

Fig. 6. The estimation error of path loss parameters with β = 1.

low complexity online channel estimation algorithms that only utilize the limited number of measurements in proximity.

Next, we examine the accuracy of the path loss parameter estimation for the scenario with unknown channel model. Assume that the channel parameters within a square area of side length D¯ is constant, and the border lines of the area is known, so that all the channel gain measurements within that region can be used for channel parameter estimation. Fig. [6](#page-7-0) compares the analytical estimation error of nPL and KdB, given in [\(23\)](#page-5-0) and [\(24\)](#page-5-1), with the simulation results, which are generated with α = 8, β = 1 and σ <sup>2</sup> = 2. It is observed that the simulation results match well with the analysis. Besides, the estimation error decreases with the region size within which the channel parameters remain constant. By comparing the CGM with different densities, we note that the estimation error can be reduced by increasing the map density in the case of β = 1.

Fig. [7](#page-7-1) compares the analytical and simulated results of path loss parameter estimation error when β = 30. In contrast of the scenario with β = 1 in Fig. [6,](#page-7-0) it is observed from Fig. [7](#page-7-1) that the reduction of the estimation error with the increase of the data collection density is negligible. This observation is consistent with our expectation, as revealed from the analytical results. Besides, while the AMSE of channel prediction depends on the distribution of the nearest sample distance, the performance of channel parameter estimation is mainly affected by the sample density. Hence, the grid and random CGM with the same density has the similar performance, as observed from both Fig. [6](#page-7-0) and Fig. [7.](#page-7-1)

Further consider the AMSE of channel prediction with the estimated parameters within a local model-consistent region.

<span id="page-7-1"></span>![](p8__page_7_Figure_10.jpeg)

Fig. 7. The estimation error of path loss parameters with β = 30.

<span id="page-7-2"></span>![](p8__page_7_Figure_12.jpeg)

Fig. 8. The AMSE of channel prediction in the local region with the estimated channel model.

When β is small, the AMSE is mainly caused by the error in path loss parameters and it changes with the number of collected data points within the region. Fig. [8\(a\)-\(b\)](#page-7-2) shows that the simulation results match well with the analytical expression shown in [\(32\).](#page-6-0) The gap between the AMSE under estimated channel parameters and that under known parameters vanishes when the number of measurements within the region is large. Fig. [8\(c\)-\(d\)](#page-7-2) shows the AMSE of channel prediction when β = 30 in a small region with N = 50 measurement samples. When the region is small, the correlated shadowing may be considered as part of the path loss. This explains why the AMSE of the channel prediction with estimated parameters is lower than that of the prediction with true channel parameters when k is small. When k gets large, the shadowing loss can be properly calculated and hence the AMSE performance converges. The performance degradation caused by parameter estimation error is almost negligible, and hence the performance can still be approximated by the analytical results given in Lemma [4.](#page-4-0)

<span id="page-8-0"></span>![](p9__page_8_Figure_2.jpeg)

Fig. 9. The AMSE of channel prediction based on grid CGM with  $d=20~\mathrm{m}$  with region division based on LoS/NLoS status.

In practical environment, finding the correct channel gain model becomes even more challenging. We consider a case study by reconstructing the CGM shown in Fig. 3(b) via the grid samples shown in Fig. 9(a) with d = 20m, where the building areas are excluded. For simplicity, the whole area is only divided into two regions based on whether the LoS link is blocked by buildings or not. Following the procedures in Section IV, the channel parameters for each region are estimated using the LS methods. For LoS region, the estimated channel parameters are  $\hat{n}_{\rm PL}=1.3442,\,\hat{K}_{\rm dB}=-77.9267,\,\hat{\alpha}=$  $0, \hat{\beta} = 0$  and  $\hat{\sigma}^2 = 9.4255$ . For NLoS region, the estimated channel parameters are  $\hat{n}_{\rm PL}=4.92,~\hat{K}_{\rm dB}=-31.6323,$  $\hat{\alpha} = 339.5941, \, \hat{\beta} = 33.9911 \, \text{and} \, \hat{\sigma}^2 = 0. \, \text{For LoS region, only}$ the path loss need to be calculated and hence the neighboring measurements are not used for online channel prediction, i.e., with k = 0. For NLoS region, based on the analytical results, k=3 is selected for balancing the accuracy and complexity for online channel prediction. The reconstructed CGM based on the estimated parameters is shown in Fig. 9(b) with the AMSE being 109.9952.

Further consider the channel prediction in a small region highlighted in Fig. 9, which contains 27 measurement samples. The estimated channel model from the samples within this region has the parameters  $\hat{n}_{\rm PL} = 0$ ,  $\hat{K}_{\rm dB} = -120.8396$ ,  $\hat{\alpha} = 148.0263$ ,  $\hat{\beta} = 7.5880$  and  $\hat{\sigma}^2 = 0$ . This best-fitting channel model looks unrealistic. This is because the distance gaps among the data collection points to the BS are very small. Hence, the channel gain difference caused by the distance gap cannot be distinguished, and the channel modelling algorithm counts the path loss into the intercept  $K_{\rm dB}$  and the correlated shadowing  $\alpha$ . Even though the estimated channel parameters are not consistent with our intuition, the model fits the measured data best, and the AMSE of the spatial prediction based on this model is reduced to 67.4203, which is much smaller than the AMSE if only the LoS/NLoS regions are considered. The performance improvement also justifies the necessity for using different models in different regions. However, if we further divide the this region into even smaller regions, as shown in Fig. 10(a), the AMSE increases since the number of data points within each region reduces. From Fig. 10(b), it is observed that if this region is further divided into two regions (separated by the y-axis in Fig. 10(a)), the AMSE slightly increases to 71.2606. If this region is

<span id="page-8-1"></span>![](p9__page_8_Figure_6.jpeg)

Fig. 10. The AMSE of channel prediction within small regions.

divided into 4 small regions, the average AMSE increases to 91.1041 and it varies in different sub-regions.

## VI. CONCLUSION

This paper investigates the CGM construction and utilization problems towards environment-aware communications. For both the random and grid distribution of data collection locations, we derive the AMSE of channel prediction as functions of the data collection density and the number of data points used for online prediction. To model the spatial variation of the wireless environment, we divide the area into small regions and estimate the local channel modeling parameters based on data points within the region. The estimation errors of the path loss parameters are derived as functions of the number of samples within the region. In general, the channel prediction error reduces with the sample density and the number of measurements used for online channel prediction, so does the AMSE. Some important lessons learnt from this study are summarized below:

- Given the channel modeling parameters, the number of measurements need to be used for online channel estimation is far below the total number of samples within the correlation distance, i.e., a small k is sufficient in most of the scenarios.
- When the correlation is strong, increasing the sample density within the region does not necessarily lead to more accurate estimation of the channel parameters, but it improves the channel prediction performance.
- For small k and large β, a larger parameter estimation error may not lead to larger channel prediction error since it is unnecessary to distinguish the correlated shadowing and path loss intercept.

Through this study, we also noticed that the proper region division is essential for model-based channel prediction. In the future, the physical map and the measured channel data will be jointly considered for region division and channel modeling. Besides, the data driven CGM construction is also promising in the complex urban environment, since it does not rely on the accuracy of channel model.

## APPENDIX A PROOF OF LEMMA 1

Since the MSE of the estimation is only related with the distance between the target location q and its nearest data

<span id="page-9-0"></span>![](p10__page_9_Picture_2.jpeg)

Fig. 11. The distribution of  $d_{\min}$  in Grid CGM.

point, we can obtain the average MSE by taking the average with respect to the distribution of  $d_{\min}$  in (10), which renders

$$\mathbb{E}[\xi_{\mathrm{dB}}(\mathbf{q})|_{(k=1)}]$$

$$= \alpha + \sigma^2 - \frac{\alpha^2}{\alpha + \sigma^2} \int_0^\infty e^{-\frac{2x}{\beta}} 2\pi \lambda x e^{-\pi \lambda x^2} dx$$

$$= \alpha + \sigma^2 - \frac{2\pi \lambda \alpha^2}{\alpha + \sigma^2} \underbrace{\int_0^\infty x e^{-\pi \lambda x^2 - \frac{2x}{\beta}} dx}_{\Xi(\lambda)}.$$

To further simplify the above equation, we have

$$\begin{split} &\Xi(\lambda) \\ &= \int_0^\infty x \exp\left(-\pi\lambda \left(x + \frac{1}{\pi\lambda\beta}\right)^2 + \frac{1}{\pi\lambda\beta^2}\right) dx \\ &= \exp\left(\frac{1}{\pi\lambda\beta^2}\right) \left(\int_{\frac{1}{\pi\lambda\beta}}^\infty x e^{-\pi\lambda x^2} dx - \frac{1}{\pi\lambda\beta} \int_{\frac{1}{\pi\lambda\beta}}^\infty e^{-\pi\lambda x^2} dx\right) \\ &\stackrel{(a)}{=} \frac{1}{2\pi\lambda} \left(1 - \frac{1}{\beta\sqrt{\lambda}} \exp\left(\frac{1}{\pi\lambda\beta^2}\right)\right) \left(1 - \Phi\left(\frac{1}{\beta\sqrt{\pi\lambda}}\right)\right) \end{split}$$

where (a) follows from the integration formulas in [22].

## APPENDIX B

## Derivation of $d_{\min}$ Distribution in Grid CGM

As shown in Fig. 11(a), the middle data point is the closest neighbor for all the target locations within the red box. Hence, finding the distribution of  $d_{\min}$  in the grid CGM, i.e.,  $P_g(x) = \Pr(d_{\min} = x)$ , is equivalent to finding the probability that a random location within the box has a distance x to the center.

From Fig. 11(b), for  $x \leq \frac{d}{2}$ , the random location has distance x with the center if it falls into the ring with inner and outer radius x and  $x+\epsilon$ , respectively, where  $\epsilon \to 0$ . Hence, we have

$$P_g(x) = \lim_{\epsilon \to 0} \frac{\pi(x+\epsilon)^2 - \pi x^2}{\epsilon d^2} = \frac{2\pi x}{d^2}.$$
 (33)

Similarly, for  $\frac{d}{2} < x < \frac{\sqrt{2}d}{2}$ , the random location has distance x from the center if it falls into the overlapped area of the ring from x and  $x + \epsilon$  and the red square. Hence, we have

$$\begin{split} P_g(x) &= \lim_{\epsilon \to 0} \frac{4x\epsilon \left(\frac{\pi}{2} - 2\arccos\left(\frac{d}{2x}\right)\right)}{\epsilon d^2} \\ &= \frac{4x}{d^2} \left(\frac{\pi}{2} - 2\arccos\left(\frac{d}{2x}\right)\right). \end{split} \tag{34}$$

<span id="page-9-1"></span>![](p10__page_9_Picture_16.jpeg)

Fig. 12. Effective measurement samples.

## APPENDIX C PROOF OF LEMMA 5

The channel parameter estimation error is given by the diagonal elements of the error covariance matrix  $C_{LS}$  in (21). Deriving the explicit expression of  $C_{LS}$  is challenging due to the inverse of a  $N \times N$  matrix  $(R_Q + \sigma^2 \mathbf{I}_{N \times N})$ . The shadowing correlation among the data points introduces the bias on the path loss parameter estimations. To get some insights on the parameter estimation error with CGM construction and utilization, we propose to group the data points that are within the correlation distance  $\beta$  and treat each group as an effective data point, as shown in Fig. 12(a). The shadowing correlation among the effective data points shown in Fig. 12(b) can be neglected, i.e., with  $R_Q = \alpha \mathbf{I}_{N \times N}$ .

Consider the random CGM with density  $\lambda$  and grid CGM with separation d. The above procedures will lead to a sparse map by a factor  $\pi\beta^2$ , i.e., on average the number of measurements grouped together is

$$c = \begin{cases} \max\{1, \pi\lambda\beta^2\}, & \text{Random CGM} \\ \max\{1, \pi\beta^2/d^2\}, & \text{Grid CGM} \end{cases}$$
 (35)

Since the multipath fading at data collection locations is assumed to be independent, the grouped data points have an effective multipath variance  $\sigma^2/c_r$ . Now, we can obtain an approximation on the parameter estimation error by considering the estimation over the effective data points after grouping, which renders

$$C_{LS} \approx (\alpha + \sigma^2/c)(\tilde{H}^T \tilde{H})^{-1} = \left(\alpha + \frac{\sigma^2}{c}\right) \begin{bmatrix} \tilde{N} & -10\log_{10} \prod_{i=1}^{\tilde{N}} d_i \\ -10\log_{10} \prod_{i=1}^{\tilde{N}} d_i & \sum_{n=1}^{\tilde{N}} (10\log_{10} d_i)^2 \end{bmatrix}^{-1},$$
(36)

where  $\tilde{H}$  contains the distance information of the effective data points,  $\tilde{N}=N/c$  is the number of effective data points,  $d_i=\|\mathbf{q}_i\|$  is the distance between the ith effective data location to the BS at the origin. We have assumed that  $d_i$  is a random variable uniformly distributed within the range  $[\delta_{\min}, \delta_{\max}]$ , then its value in dB scale, denoted by  $y_i=10\log_{10}d_i$ , is distributed according to

$$f(y) = \Pr(y_i = y) = \frac{\ln(10)}{(\delta_{\text{max}} - \delta_{\text{min}})} 10^{(\frac{y}{10} - 1)}.$$
 (37)

Denote by  $\mu=\frac{1}{\tilde{N}}\sum_{i=1}^{\tilde{N}}y_i$ . When  $\tilde{N}$  is sufficiently large,  $\mu$  will approach the expectation of  $y_i$ , which is derived as

$$\begin{split} \mu &= \int_{10 \log_{10} \delta_{\text{max}}}^{10 \log_{10} \delta_{\text{min}}} y f(y) dy \\ &= \frac{10 \delta_{\text{max}} \log_{10} (\delta_{\text{max}}) - 10 \delta_{\text{min}} \log_{10} (\delta_{\text{min}})}{\delta_{\text{max}} - \delta_{\text{min}}} - \frac{10}{\ln 10} \end{split}$$
(38)

Denoted by  $\chi$  the variance of the random variable  $y_i$ , we have

$$\chi = \int_{10 \log_{10} \delta_{\min}}^{10 \log_{10} \delta_{\min}} (y - \mu)^{2} f(y) dy$$

$$= \frac{100}{(\ln 10)^{2}} - \frac{100 \delta_{\max} \delta_{\min} \log_{10} (\delta_{\max} / \delta_{\min})}{(\delta_{\max} - \delta_{\min})^{2}}.$$
(39)

When  $\tilde{N}$  is sufficiently large, we have  $\sum_{n=1}^N (10\log_{10}d_i)^2 = \sum_{n=1}^{\tilde{N}}y_i^2 = \tilde{N}(\chi+\mu^2)$ . Hence the error covariance matrix can be in (36) can be written as

$$C_{LS} = (\alpha + \sigma^2/c) \begin{bmatrix} \tilde{N} & -\tilde{N}\mu \\ -\tilde{N}\mu & \tilde{N}(\chi + \mu^2) \end{bmatrix}^{-1}$$
$$= \frac{\alpha + \sigma^2/c}{\tilde{N}\gamma} \begin{bmatrix} \chi + \mu^2 & \mu \\ \mu & 1 \end{bmatrix}$$
(40)

Together with (22), it completes the proof of Lemma 5.

## REFERENCES

- Y. Zeng et al., "A tutorial on environment-aware communications via channel knowledge map for 6G," *IEEE Commun. Surveys Tuts.*, early access, Feb. 9, 2024, doi: 10.1109/COMST.2024.3364508.
- [2] Y. Zeng and X. Xu, "Toward environment-aware 6G communications via channel knowledge map," *IEEE Wireless Commun.*, vol. 28, no. 3, pp. 84–91, Jun. 2021.
- [3] D. Wu, Y. Zeng, S. Jin, and R. Zhang, "Environment-aware hybrid beamforming by leveraging channel knowledge map," *IEEE Trans. Wireless Commun.*, early access, Oct. 18, 2023, doi: 10.1109/TWC.2023.3323941.
- [4] Study on 3D Channel Model for LTE, Standard 3GPP TR 36.873, 2017.
- [5] M. Malmirchegini and Y. Mostofi, "On the spatial predictability of communication channels," *IEEE Trans. Wireless Commun.*, vol. 11, no. 3, pp. 964–978, Mar. 2012.
- [6] J. Thrane, D. Zibar, and H. L. Christiansen, "Model-aided deep learning method for path loss prediction in mobile communication systems at 2.6 GHz," *IEEE Access*, vol. 8, pp. 7925–7936, 2020.
- [7] W. Liu and J. Chen, "UAV-aided radio map construction exploiting environment semantics," *IEEE Trans. Wireless Commun.*, vol. 22, no. 9, pp. 6341–6355, Jun. 2023.
- [8] V.-P. Chowdappa, C. Botella, J. J. Samper-Zapater, and R. J. Martinez, "Distributed radio map reconstruction for 5G automotive," *IEEE Intell. Transp. Syst. Mag.*, vol. 10, no. 2, pp. 36–49, Summer. 2018.
- [9] K. Sato and T. Fujii, "Kriging-based interference power constraint: Integrated design of the radio environment map and transmission power," *IEEE Trans. Cogn. Commun. Netw.*, vol. 3, no. 1, pp. 13–25, Mar. 2017.
- [10] A. Verdin, C. Funk, B. Rajagopalan, and W. Kleiber, "Kriging and local polynomial methods for blending satellite-derived and gauge precipitation estimates to support hydrologic early warning systems," *IEEE Trans. Geosci. Remote Sens.*, vol. 54, no. 5, pp. 2552–2562, May 2016.
- [11] K. Li, P. Li, Y. Zeng, and J. Xu, "Channel knowledge map for environment-aware communications: EM algorithm for map construction," in *Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)*, Apr. 2022, pp. 1659–1664.
- [12] M. Gudmundson, "Correlation model for shadow fading in mobile radio systems," *Electron. Lett.*, vol. 27, no. 23, p. 2145, 1991.
- [13] D. S. Kalogerias and A. P. Petropulu, "Nonlinear spatiotemporal channel gain map tracking in mobile cooperative networks," in *Proc. IEEE 16th Int. Workshop Signal Process. Adv. Wireless Commun. (SPAWC)*, Jun. 2015, pp. 660–664.

- [14] Q. Liao, S. Valentin, and S. Stanczak, "Channel gain prediction in wireless networks based on spatial-temporal correlation," in *Proc. IEEE* 16th Int. Workshop Signal Process. Adv. Wireless Commun. (SPAWC), Jun. 2015, pp. 400–404.
- [15] L. S. Muppirisetty, T. Svensson, and H. Wymeersch, "Spatial wireless channel prediction under location uncertainty," *IEEE Trans. Wireless Commun.*, vol. 15, no. 2, pp. 1031–1044, Feb. 2016.
- [16] N. Cressie, Statistics for Spatial Data. Hoboken, NJ, USA: Wiley, 2015.
- [17] A. Chaves-Villota and C. A. Viteri-Mera, "DeepREM: Deep-learning based radio environment map estimation from sparse measurements," *IEEE Access*, vol. 11, pp. 48697–48714, May 2023, doi: 10.1109/ACCESS.2023.3277248.
- [18] E. Krijestorac, S. Hanna, and D. Cabric, "Spatial signal strength prediction using 3D maps and deep learning," in *Proc. ICC IEEE Int. Conf. Commun.*, Jun. 2021, pp. 1–6.
- [19] A. Karttunen, A. F. Molisch, S. Hur, J. Park, and C. J. Zhang, "Spatially consistent street-by-street path loss model for 28-GHz channels in micro cell urban environments," *IEEE Trans. Wireless Commun.*, vol. 16, no. 11, pp. 7538–7550, Nov. 2017.
- [20] W. S. K. S. N. Chiu, D. Stoyan, and J. Mecke, Stochastic Geometry and its Applications, 3rd ed. Hoboken, NJ, USA: Wiley, 2013.
- [21] N. Cressie, "Fitting variogram models by weighted least squares," J. Int. Assoc. Math. Geol., vol. 17, no. 5, pp. 563–586, Jul. 1985.
- [22] I. S. Gradshteyn and I. M. Ryzhik, Table of Integrals, Series and Products, 7th ed. Amsterdam, The Netherlands: Elsevier, 2007.

![](p11__page_10_Picture_32.jpeg)

Xiaoli Xu (Member, IEEE) received the Bachelor of Engineering (Hons.) and Ph.D. degrees from Nanyang Technological University, Singapore, in 2009 and 2015, respectively. From 2015 to 2018, she was a Research Fellow with Nanyang Technological University. From 2018 to 2019, she was a Post-Doctoral Research Associate with The University of Sydney, Australia. She is currently with the School of Information Science and Engineering. Southeast University, China. Her research interests include network coding, information theory, vehicu-

lar ad-hoc networks, and channel knowledge map (CKM). She received the Best Paper Award for the 10th IEEE International Conference on Information, Communications and Signal Processing.

![](p11__page_10_Picture_35.jpeg)

Yong Zeng (Senior Member, IEEE) received the Bachelor of Engineering (Hons.) and Ph.D. degrees from Nanyang Technological University, Singapore. From 2013 to 2018, he was a Research Fellow and a Senior Research Fellow with the Department of Electrical and Computer Engineering, National University of Singapore. From 2018 to 2019, he was a Lecturer with the School of Electrical and Information Engineering, The University of Sydney, Australia. He is currently with the National Mobile Communications Research Laboratory, Southeast

University, China, and Purple Mountain Laboratories, Nanjing, China. He has published more than 170 articles, which have been cited by more than 25,000 times based on Google Scholar. He was listed as a Highly Cited Researcher by Clarivate Analytics for five consecutive years from 2019 to 2023. He was a recipient of Australia Research Council (ARC) Discovery Early Career Researcher Award (DECRA), the 2020 and 2024 Marconi Prize Paper Award in Wireless Communications, the 2018 IEEE Communications Society Asia-Pacific Outstanding Young Researcher Award, the 2020 and 2017 IEEE Communications Society Heinrich Hertz Prize Paper Award, the 2021 IEEE ICC Best Paper Award, and the 2021 China Communications Best Paper Award. He is the Symposium Chair of the IEEE GLOBECOM 2021 Track on Aerial Communications, the Workshop Co-Chair of ICC 2018-2023 Workshop on UAV Communications, and the Tutorial Speaker of GLOBECOM 2018/2019 and ICC 2019 Tutorials on UAV Communications. He serves as an Editor for IEEE TRANSACTIONS ON COMMUNICATIONS, IEEE COMMUNICATIONS LETTERS, and IEEE OPEN JOURNAL OF VEHICULAR TECHNOLOGY, and the Leading Guest Editor for IEEE WIRELESS COM-MUNICATIONS on "Integrating UAVs into 5G and Beyond" and CHINA COMMUNICATIONS on "Network-Connected UAV Communications."