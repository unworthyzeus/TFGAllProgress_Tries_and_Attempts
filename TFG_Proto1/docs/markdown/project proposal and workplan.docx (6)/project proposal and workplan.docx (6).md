![](p1__page_0_Picture_0.jpeg)

## Channel Knowledge Map Prediction with Deep Learning for 6G UAV-enabled Networks

Guillem Moreno Garcia

Project Proposal and Work Plan

| Document:        | [project | proposal | and |  |  |
|------------------|----------|----------|-----|--|--|
| workplan.doc]    |          |          |     |  |  |
| Date: 04/03/2026 |          |          |     |  |  |
| Rev: 01          |          |          |     |  |  |
| Page 2 of 13     |          |          |     |  |  |

## Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p2__page_1_Picture_2.jpeg)

## REVISION HISTORY AND APPROVAL RECORD

| Revision | Date       | Purpose                   |
|----------|------------|---------------------------|
| 0        | 16/02/2026 | Document<br>creation      |
| 1        | 26/02/2026 | Document<br>revision<br>1 |
| 2        | 27/02/2026 | Document<br>revision 2    |
| 3        | 01/03/2026 | Document revision 3       |
| 4        | 02/03/2026 | Document revision 4       |
| 5        | 04/03/2026 | Document submission       |
|          |            |                           |
|          |            |                           |
|          |            |                           |

## DOCUMENT DISTRIBUTION LIST

| Name                  | E-mail                                    |
|-----------------------|-------------------------------------------|
| Guillem<br>Moreno     | guillem.moreno.garcia@estudiantat.upc.edu |
|                       |                                           |
| Sergi<br>Abadal       | sergi.abadal@upc.edu                      |
| Evgenii<br>Vinogradov | evgenii.vinogradov@upc.edu                |
|                       |                                           |
|                       |                                           |
|                       |                                           |
|                       |                                           |
|                       |                                           |

| WRITTEN<br>BY: |                             | REVIEWED<br>AND<br>APPROVED<br>BY: |                       |
|----------------|-----------------------------|------------------------------------|-----------------------|
|                |                             |                                    |                       |
|                |                             |                                    |                       |
| Date           | 16/2/2026                   | Date                               | 02/03/2026            |
| Name           | Guillem<br>Moreno<br>Garcia | Name                               | Sergi<br>Abadal       |
| Position       | Project<br>author           | Position                           | Project<br>Supervisor |

| Document:<br>workplan.doc] | [project | proposal | and |  |
|----------------------------|----------|----------|-----|--|
| Date: 04/03/2026           |          |          |     |  |
| Rev: 01                    |          |          |     |  |
|                            |          |          |     |  |

Page 3 of 13

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV**

**enabled Networks**

![](p3__page_2_Picture_2.jpeg)

<span id="page-2-0"></span>

| 0. | CONTENTS                            |                                                 |   |
|----|-------------------------------------|-------------------------------------------------|---|
| 0. | Contents                            |                                                 | 3 |
| 1. | Project<br>overview<br>and<br>goals |                                                 | 4 |
| 2. | Project                             | background                                      | 5 |
| 3. | Project                             | requirements<br>and<br>specifications           | 6 |
| 4. | Work                                | Plan                                            | 7 |
|    | 4.1.                                | Work<br>Breakdown<br>Structure                  | 7 |
|    | 4.2.                                | Work<br>Packages,<br>Tasks<br>and<br>Milestones | 7 |
|    | 4.3.                                | Time<br>Plan<br>(Gantt<br>diagram)              | 8 |
|    | 4.4.                                | Meeting<br>and<br>communication<br>plan         | 9 |

5. [Generic](#page-12-0) skills 10

Document: [project proposal and workplan.doc] Date: 04/03/2026 Rev: 01 Page 4 of 13

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p4__page_3_Picture_2.jpeg)

## **1. PROJECT OVERVIEW AND GOALS**

This project is carried out at the Nanonetworking Center in Catalonia (N3Cat), part of the Broadband Communications Systems and Architectures Research Group (CBA) within the Department of Computer Architecture.

## **Summary of the subject:**

The deployment of 6G networks introduces a paradigm shift towards environment-aware communications, heavily integrating non-terrestrial nodes such as Uncrewed Aerial Vehicles (UAVs). This transition is accompanied by an industry-wide focus on the FR3 upper-midband spectrum (7.125 GHz, defined as the primary test band in the EU). To optimize these networks without the prohibitive computational overhead of continuous real-time channel estimation, it is critical to proactively predict how radio waves interact with complex 3D urban environments using accurate Channel Knowledge Maps (CKMs). Traditionally, this deterministic modeling is achieved using Ray Tracing (RT) software. While RT provides exceptional physical accuracy, it is extremely computationally heavy and slow. For highly dynamic 3D networks involving mobile UAVs with varying antenna deployment heights, running RT simulations for every possible spatial configuration creates a severe bottleneck. This limits practical scalability and is entirely incompatible with the real-time demands of 6G.

To overcome this limitation, recent literature has focused on data-driven Artificial Intelligence (AI) and Machine Learning (ML) approaches capable of predicting channel parameters directly from environmental geometry. Addressing the RT bottleneck, this project builds upon these foundations by proposing a standalone ML architecture that frames 3D channel prediction as an advanced image-to-image translation problem. By leveraging massive research-grade 3D city datasets (e.g., GlobalBuildingAtlas and 3DglobFP), the model will learn to seamlessly map physical environments directly to radio channel characteristics.

The development pipeline will initially leverage publicly available 5G datasets to establish a methodological baseline for environment-aware Convolutional Neural Networks (CNNs). Subsequently, the project will transition to a proprietary 6G FR3 dataset generated internally by the N3Cat research group.

## **The project main goals are:**

1. Develop a standalone deep learning software architecture capable of predicting key 6G FR3 channel parameters (delay spread, angular spread, and channel power) from 2D ground representation and varying UAV altitudes (up to a maximum height of approximately 500 meters).

| Document:        | [project | proposal | and |
|------------------|----------|----------|-----|
| workplan.doc]    |          |          |     |
| Date: 04/03/2026 |          |          |     |
| Rev: 01          |          |          |     |
| Page 5 of 13     |          |          |     |

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p5__page_4_Picture_2.jpeg)

- 2. Accelerate channel generation by achieving a 10x to 100x reduction in computation time compared to traditional ray tracing simulations.
- 3. Attain high prediction accuracy against ground-truth data, targeting a Root Mean Square Error (RMSE) of at most approximately 3 to 5 dB for channel power, 50 ns for delay spread, and 20 degrees for angular spread.
- 4. Implement data augmentation pipelines to compensate for the computational expense of generating massive ground-truth datasets, ensuring the model generalizes well across diverse global urban layouts and good heuristics that make sure that the data outputted by the model makes sense with deterministic known formulas for 6G UAV Enabled networks.

| Document:        | [project | proposal | and |
|------------------|----------|----------|-----|
| workplan.doc]    |          |          |     |
| Date: 04/03/2026 |          |          |     |
| Rev: 01          |          |          |     |
| Page 6 of 13     |          |          |     |

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p6__page_5_Picture_2.jpeg)

## **2. PROJECT BACKGROUND**

This project is performed within the framework of the research activities conducted at the Nanonetworking Center in Catalonia (N3Cat). Furthermore, this work stems from the research activities supported by the grant INVESTIGADOR/A POSTDOCTORAL-RAMÓN Y CAJAL, RYC2024-051003-I, funded by MICIU/AEI/10.13039/501100011033 and by the FSE+ (E. Vinogradov is a PI).

While the specific deep learning software architecture, data augmentation pipelines, and model evaluations for this Bachelor's thesis are being developed from scratch by the author, the project strongly leverages foundational methodologies and proprietary 6G FR3 datasets generated by the N3Cat research group.

The main initial ideas for the project (i.e., the conceptualization of framing 6G FR3 channel prediction as an image-to-tensor translation problem using 3D city models and the objective to accelerate channel knowledge map generation for UAV networks) were provided by the project supervisors (Evgenii Vinogradov and Sergi Abadal). The project author is responsible for the autonomous research, design, implementation, and rigorous testing of the machine learning architecture proposed to fulfill these objectives.

| Document:        | [project | proposal | and |
|------------------|----------|----------|-----|
| workplan.doc]    |          |          |     |
| Date: 04/03/2026 |          |          |     |
| Rev: 01          |          |          |     |
| Page 7 of 13     |          |          |     |

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p7__page_6_Picture_2.jpeg)

## **3. PROJECT REQUIREMENTS AND SPECIFICATIONS**

## Project requirements:

- Predict delay spread, angular spread, channel power and augmented line of sight for a whole city map based on a deep learning model.
- This will be for 6G FR3 UAV-enabled networks.

## Project specifications:

- Predict delay spread, angular spread, channel power and augmented line of sight based on a height matrix, Z value of the antenna (always in the middle of the image), with a frequency of 7.125GHz which corresponds to the FR3 6G frequency band and a binary line of sight map (this last one optional)
- 3D model of a city -> grayscale images (one matrix or two if it has the binary line of sight) + height of the antenna per image -> output tensors (one per image) with delay Spread, angular spread, channel power and maybe the augmented line of sight.

| Document:<br>workplan.doc] | [project | proposal | and |
|----------------------------|----------|----------|-----|
| Date: 04/03/2026           |          |          |     |
| Rev: 01                    |          |          |     |
| Page 8 of 13               |          |          |     |

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p8__page_7_Picture_2.jpeg)

## **4. WORK PLAN**

# 4.1. *Work Breakdown Structure*

(Work packages breakdown diagram)

![](p8__page_7_Figure_7.jpeg)

# 4.2. *Work Packages, Tasks and Milestones*

### Work Packages:

| Project:<br>Channel<br>Knowledge<br>Map<br>Prediction<br>with<br>Deep<br>Learning<br>for<br>6G<br>UAV-enabledNetworks | WP<br>ref:<br>WP1                       |
|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| Major<br>constituent:<br>Introduction<br>and<br>Research                                                              | Sheet<br>1<br>of<br>5                   |
| Short<br>description:                                                                                                 | Planned<br>start<br>date:<br>20/01/2026 |
| Researching<br>the<br>formulas<br>for<br>the<br>channel<br>knowledge<br>map<br>for                                    | Planned<br>end<br>date:<br>15/03/2026   |
| channel<br>power,<br>angular<br>spread,<br>delay<br>spread<br>and<br>augmented                                        |                                         |

| Document:<br>workplan.doc] | [project | proposal | and |
|----------------------------|----------|----------|-----|
| Date: 04/03/2026           |          |          |     |
| Rev: 01                    |          |          |     |
|                            |          |          |     |

Page 9 of 13

## Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p9__page_8_Picture_2.jpeg)

| line<br>of<br>sight.<br>Also<br>research<br>the<br>state<br>of<br>the<br>art<br>for<br>image<br>generation<br>(types<br>of<br>CNNs)<br>based<br>on<br>environment-aware<br>parameters,<br>specifically<br>communications.<br>See<br>what<br>has<br>been<br>done<br>for<br>5G<br>ground<br>based<br>antenna<br>and<br>see<br>the<br>differences<br>that<br>there<br>will<br>be<br>6G<br>UAV<br>Enabled<br>Networks. | Start<br>event:<br>T1.1<br>End<br>event:<br>T1.3 |        |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|--------|
| Internal<br>task<br>T1.1:<br>Research<br>the<br>best<br>parameters<br>for<br>simulations<br>of<br>diverse<br>data<br>of<br>6G<br>UAV<br>Enabled<br>Networks<br>to<br>then<br>train<br>the<br>Neural<br>Network.                                                                                                                                                                                                    | Deliverables:                                    | Dates: |
| Internal<br>task<br>T1.2:<br>Research<br>the<br>formulas<br>that<br>apply<br>for<br>6G<br>UAV<br>Enabled<br>Networks                                                                                                                                                                                                                                                                                               |                                                  |        |
| Internal<br>Task<br>T1.3:<br>Do<br>a<br>dummy<br>training<br>of<br>some<br>5G<br>dataset<br>to<br>learn<br>what<br>are<br>the<br>best<br>networks<br>and<br>how<br>to<br>heuristically<br>improve<br>the<br>results.                                                                                                                                                                                               |                                                  |        |

| Project:<br>Channel<br>Knowledge<br>Map<br>Prediction<br>with<br>Deep<br>Learning<br>for<br>6G<br>UAV-enabled<br>Networks                                                                                | WP<br>ref:<br>WP2                       |        |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|--------|
| Major<br>constituent:<br>Data<br>augmentation<br>and<br>heuristics                                                                                                                                       | Sheet<br>2<br>of<br>5                   |        |
| Short<br>description:<br>The<br>simulations<br>of<br>the<br>ground<br>truth<br>data<br>will                                                                                                              | Planned<br>start<br>date:<br>16/03/2026 |        |
| be<br>in<br>progress.<br>I'll<br>have<br>access<br>to<br>the<br>ground<br>truth<br>of<br>the<br>cities,<br>but<br>not<br>enough<br>to<br>train<br>the<br>full<br>model.<br>But<br>train<br>some<br>dummy | Planned<br>end<br>date:<br>20/04/2026   |        |
| models<br>(for<br>each<br>output)<br>to<br>see<br>what<br>kind<br>of<br>data<br>augmentation                                                                                                             | Start<br>event:<br>T2.1                 |        |
| will<br>be<br>needed<br>for<br>the<br>best<br>results.<br>Based<br>on<br>the<br>research<br>of<br>the                                                                                                    | End<br>event:<br>T2.2                   |        |
| ongoing<br>WP1<br>I<br>can<br>say<br>that<br>cGANs<br>will<br>probably<br>be<br>used.                                                                                                                    |                                         |        |
| cGANs<br>(Conditional<br>Generative<br>Adversarial<br>Networks)<br>are                                                                                                                                   |                                         |        |
| machine<br>learning<br>models<br>capable<br>of<br>generating<br>new,<br>realistic                                                                                                                        |                                         |        |
| data<br>constrained<br>by<br>specific<br>inputs.<br>In<br>this<br>project,<br>they<br>will<br>likely                                                                                                     |                                         |        |
| be<br>used<br>to<br>simulate<br>data<br>based<br>on<br>our<br>3D<br>environments.                                                                                                                        |                                         |        |
| Internal<br>task<br>T2.1:<br>Data<br>augmentation<br>pipelines<br>to<br>get<br>the<br>best<br>results<br>from<br>deep<br>learning                                                                        | Deliverables:                           | Dates: |
|                                                                                                                                                                                                          |                                         |        |
| Internal<br>task<br>T2.2:<br>Researching<br>heuristics<br>to<br>apply<br>apart<br>from<br>the<br>model<br>output<br>to<br>improve<br>the<br>results.                                                     |                                         |        |

| Project:<br>Channel<br>Knowledge<br>Map<br>Prediction<br>with<br>Deep<br>Learning                                                                                                    | WP<br>ref:<br>WP3                       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| for<br>6G<br>UAV-enabled<br>Networks                                                                                                                                                 |                                         |
| Major<br>constituent:<br>Deep<br>learning<br>model<br>and<br>software<br>pipeline                                                                                                    | Sheet<br>3<br>of<br>5                   |
| Short<br>description:<br>Based<br>on<br>the<br>results<br>obtained<br>on<br>the<br>first<br>2                                                                                        | Planned<br>start<br>date:<br>21/04/2026 |
| WPs,<br>train<br>the<br>best<br>model<br>possible<br>for<br>channel<br>knowledge<br>map<br>prediction<br>for<br>6G<br>UAV-enabled<br>networks,<br>at<br>the<br>start<br>we<br>should | Planned<br>end<br>date:<br>07/06/2026   |
| have<br>decided<br>if<br>we<br>want<br>the<br>binary<br>line<br>of<br>sight<br>input<br>or<br>not<br>and                                                                             | Start<br>event:<br>T3.1                 |
| the<br>augmented<br>line<br>of<br>sight<br>or<br>not<br>as<br>an<br>output.<br>Also,<br>if<br>required,                                                                              | End<br>event:<br>T3.2                   |
| to<br>algorithmically<br>get<br>the<br>line<br>of<br>sight<br>from<br>images<br>and<br>antenna                                                                                       |                                         |

## Document: [project proposal and workplan.doc] Date: 04/03/2026 Rev: 01

Page 10 of 13

## Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p10__page_9_Picture_2.jpeg)

| heights.<br>Also<br>maybe<br>include<br>actually<br>obtaining<br>the<br>binary<br>images/matrices<br>from<br>the<br>3D<br>maps.            |               |        |
|--------------------------------------------------------------------------------------------------------------------------------------------|---------------|--------|
| Internal<br>task<br>T3.1:<br>Deep<br>learning<br>model<br>Internal<br>task<br>T3.2:<br>Software<br>pipeline<br>to<br>get<br>the<br>results | Deliverables: | Dates: |

| Project:<br>Channel<br>Knowledge<br>Map<br>Prediction<br>with<br>Deep<br>Learning<br>for<br>6G<br>UAV-enabled<br>Networks                                                                                | WP<br>ref:<br>WP4                                                                |        |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|--------|
| Major<br>constituent:<br>Testing                                                                                                                                                                         | Sheet<br>4<br>of<br>5                                                            |        |
| Short<br>description:<br>The<br>results<br>will<br>be<br>compared<br>in<br>precision<br>to<br>the<br>ground<br>truth<br>and<br>see<br>in<br>which<br>cities<br>and<br>parameters<br>it<br>fails<br>more. | Planned<br>start<br>date:<br>08/06/2026<br>Planned<br>end<br>date:<br>18/06/2026 |        |
|                                                                                                                                                                                                          | Start<br>event:<br>start<br>of<br>T4.1<br>End<br>event:<br>end<br>of<br>T4.1     |        |
| Internal<br>task<br>T4.1:<br>Test<br>the<br>results                                                                                                                                                      | Deliverables:                                                                    | Dates: |

| Project:<br>Channel<br>Knowledge<br>Map<br>Prediction<br>with<br>Deep<br>Learning<br>for<br>6G<br>UAV-enabled<br>Networks                                                                                                                                                      | WP<br>ref:<br>WP5                                                                                                                    |        |  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------|--|
| Major<br>constituent:<br>Project<br>Report                                                                                                                                                                                                                                     | Sheet<br>5<br>of<br>5                                                                                                                |        |  |
| Short<br>description:<br>After<br>getting<br>the<br>final<br>results<br>and<br>best<br>model<br>possible<br>I'll<br>write<br>the<br>project<br>report<br>(bachelor<br>thesis)<br>with<br>the<br>correct<br>citations,<br>problems<br>encountered<br>and<br>final<br>precision. | Planned<br>start<br>date:<br>16/03/2026<br>Planned<br>end<br>date:<br>30/06/2026<br>Start<br>event:<br>T5.1<br>End<br>event:<br>T5.2 |        |  |
| Internal<br>task<br>T5.1:<br>Project<br>Report<br>writing<br>Internal<br>task<br>T5.2:<br>Presentation<br>preparation.                                                                                                                                                         | Deliverables:                                                                                                                        | Dates: |  |

## **Milestones**

| WP# | Task# | Short title | Milestone / deliverable | Date |
|-----|-------|-------------|-------------------------|------|
|     |       |             |                         |      |

| Document:        | [project | proposal | and |  |  |  |
|------------------|----------|----------|-----|--|--|--|
| workplan.doc]    |          |          |     |  |  |  |
| Date: 04/03/2026 |          |          |     |  |  |  |
| Rev: 01          |          |          |     |  |  |  |

Page 11 of 13

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p11__page_10_Picture_2.jpeg)

| WP1 | T1.1,<br>T1.2,<br>T1.3 | WorkPlan Approval               | Project Proposal and WorkPlan approval                  | 05/03/2026 |
|-----|------------------------|---------------------------------|---------------------------------------------------------|------------|
| WP5 | T5.1                   | Midterm Review                  | Critical Review (midterm)                               | 02/04/2026 |
| WP2 | T2.1,<br>T2.2          | Data augmentation<br>completion | Data augmentation pipelines and<br>heuristics finalized | 20/04/2026 |
| WP3 | T3.1,<br>T3.2,<br>T3.3 | Pipeline Completion             | Deep learning model and pipeline<br>finalized           | 07/06/2026 |
| WP4 | T4.1                   | Testing Completion              | Precision results compared to ground<br>truth           | 18/06/2026 |
| WP5 | T5.1                   | Final Review                    | Final Review meeting                                    | 18/06/2026 |
| WP5 | T5.1                   | Project Delivery                | Final thesis report                                     | 21/06/2026 |
| WP5 | T5.2                   | Presentation                    | Final presentation                                      | 01/07/2026 |

| Document:        | [project | proposal | and |
|------------------|----------|----------|-----|
| workplan.doc]    |          |          |     |
| Date: 04/03/2026 |          |          |     |
| Rev: 01          |          |          |     |
| Page 12 of 13    |          |          |     |

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p12__page_11_Picture_2.jpeg)

## 4.3. *Time Plan (Gantt diagram)*

![](p12__page_11_Figure_4.jpeg)

## 4.4. *Meeting and communication plan*

- Planned meetings with the supervisor:

| Date                     |
|--------------------------|
| 5th<br>of<br>March       |
|                          |
| 2th<br>of<br>April       |
|                          |
| 18th<br>of<br>June       |
|                          |
| Weekly<br>on<br>Thursday |
|                          |

| Document:        | [project | proposal | and |
|------------------|----------|----------|-----|
| workplan.doc]    |          |          |     |
| Date: 04/03/2026 |          |          |     |
| Rev: 01          |          |          |     |
| Page 13 of 13    |          |          |     |

Project Proposal and Work Plan **Channel Knowledge Map Prediction with Deep Learning for 6G UAV enabled Networks**

![](p13__page_12_Picture_2.jpeg)

## **5. GENERIC SKILLS**

The following generic skills will be promoted and assessed during the development of the project: (Mark at least three, being GS4 one of them)

Be aware that if you have some of the third level generic skills not scored yet with A or B, you can work them in your TFG in order to obtain your Bachelor degree with the set of generic skills completely acquired.

| #  | Generic<br>Skill                                                                                                          | Assessed |
|----|---------------------------------------------------------------------------------------------------------------------------|----------|
| 1  | Innovation<br>and<br>entrepreneurship                                                                                     |          |
| 2  | Societal<br>and<br>environmental<br>context                                                                               |          |
| 3  | Communication<br>in<br>a<br>foreign<br>language                                                                           | X        |
| 4  | Oral<br>and<br>written<br>communication                                                                                   | X        |
| 5  | Teamwork                                                                                                                  |          |
| 6  | Survey<br>of<br>information<br>resources                                                                                  | X        |
| 7  | Autonomous<br>learning                                                                                                    | X        |
| 8  | Ability<br>to<br>identify,<br>formulate<br>and<br>solve<br>engineering<br>problems                                        | X        |
| 9  | Ability<br>to<br>Conceive,<br>Design,<br>Implement<br>and<br>Operate<br>complex<br>systems<br>in<br>the<br>ICT<br>context | X        |
| 10 | Experimental<br>behaviour<br>and<br>ability<br>to<br>manage<br>instruments                                                |          |