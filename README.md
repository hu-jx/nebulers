# nebulers
We have chosen the following three subsystems to focus on in resolving the problems stated in Problem Statement 3

### **Structural Health Monitoring**

Carbodies and bogie frames are safety-critical structures. Undetected fatigue accumulation could eventually cause cracks or structural failure, potentially producing severe consequences across the whole vehicle.

Its prediction also supports long-term preventive maintenance: maintenance can be scheduled from cumulative damage and remaining-life estimates rather than waiting for visible deterioration. Although structural failures are comparatively rare, their severity makes this subsystem the most important from a risk perspective.

This system predicts a single numeric cumulative fatigue damage value for each train dynamic-stress time series. 

#### Approach

Extract statistical, dynamic and fatigue-related features:

* Statistical features such as the mean, standard deviation, minimum, maximum, range, percentiles, RMS, and mean absolute stress.  
* Dynamic features, which describe changes between consecutive stress readings.   
* Identify stress cycles using rainflow cycle counting, from which the total number of cycles, maximum cycle range, weighted mean cycle range, and a Miner’s-rule-inspired pseudo-damage feature are calculated.

A Random Forest Regressor was trained using the aforementioned features.

* Both raw and logarithmically transformed damage targets were considered, with the log transformation helping reduce the influence of highly skewed damage values.   
* GridSearchCV with five-fold cross-validation is used to select suitable model hyperparameters.   
* The model is evaluated using a train-test split and leave-one-out cross-validation.  
  * LOO-CV was used as an additional diagnostic. 

After model selection, the final model is retrained using all labelled training data and saved. The predict.py script loads this model, processes unseen test files using the same feature-extraction procedure, and generates shm\_predictions.csv. 

### **Rail corrugation diagnosis**

Rail corrugation directly affects wheel–rail interaction. If it develops unchecked, it can:

* increase vibration and dynamic wheel–rail forces,  
* accelerate damage to fasteners, track components, and vehicle running gear,  
* reduce ride stability and comfort,  
* increase noise,  
* contribute to derailment risk in severe cases.

It may occur more frequently than serious structural fatigue failures and can affect many trains passing over the same track section. I rank it second because its system-wide impact is substantial, though corrugation is usually a progressive maintenance problem rather than an immediate catastrophic failure.

#### Approach

The solution uses a physics-informed pipeline to identify rail corrugation from the vibration and shock signals recorded across both rails. Each 1-second recording is transformed into features that capture the periodic and side-specific characteristics of rail corrugation, while accounting for changes in operating speed.

Feature engineering

* Frequency-domain energy, speed-normalised wavelength features and per-car Side I and Side II asymmetry are extracted from the vibration signals.   
* Narrowband peak concentration, spectral entropy and cross-car variation are also included to capture the repeated spectral structure associated with rail corrugation

Model selection

* A class-balanced RBF SVM and weighted XGBoost classifier are combined through logistic regression stacking. The two models capture different structure in the engineered feature space: the SVM handles non-linear class boundaries, while XGBoost models interactions across spectral, wavelength and side-specific features.  
* The stacked design preserves the physics-informed representation while combining complementary decision behaviour from both models.   
* This hybrid configuration gave the strongest validation performance among the tested approaches, including side-wise and hierarchical classifiers such as MiniROCKET and a 1D CNN.

Evaluation

* Macro F1 is used because the class distribution is highly imbalanced and performance on Side I and Side II must carry the same weight as Normal. The final configuration achieved a mean Macro F1 of 0.746 across 5 x 5 repeated stratified cross-validation.

Assumptions made

* Each recording is treated as one independent sample and assigned one label (Normal, Side I, Side II).  
* Train speed is only used to convert frequency into wavelength and define the relevant spectral region; it is not included directly as a predictive feature. 

### **Door fault diagnosis**

Door faults have a direct passenger-facing safety impact. Door faults are also operationally frequent and highly visible.

Abnormal resistance can lead to:

* incomplete opening or closing,  
* passenger or object trapping,  
* motor overload,  
* doors becoming unavailable,  
* service delays or train withdrawal.

#### Approach 

The problem was divided into two separate parts: Segmentation and Classification. Classification built up on the segments that were identified in segmentation to allow for greater clarity of not just the individual timestamp values but the entire cycle as a whole – since a fault would occur through the entire cycle. 

Segmentation 

* Timestamp gaps were used as the baseline segmentation signal. This is due to a significant and clean gap found between cycles, allowing for ease of utilising this signal as a way of segmenting cycles.   
* Timestamp gap thresholds are determined from the training data, validated with the training data answers and freezed. The thresholds are then applied onto other datasets for segmentation into cycles.   
* The segmentation method achieved 100% accuracy, in a held-out validation dataset from the training data and verified with the train segment answers.   
* However, timestamp gaps while being a clean and significant signal here, may not persist through other cycles in the same way as well. A key limitation of our segmentation method is in the lack of signal-based verification and segmentation given the lack of time.   
* While timestamp gaps remain a dominant signal based on training data, having signal-based segmentation (i.e. using other signals like DCSR, DCSL) to verify the identified segments and iteratively adjust segment start and end times would benefit the segmentation algorithm.

Classification

* During AI-assisted exploratory data analysis, current was found to be a significant signal of abnormal resistance. Peak mean current in abnormal resistance would differ from that of normality in terms of the magnitude of the peak. The phase in which the peak occurred however, differed based on the type of operation (Open/Close) being completed.   
* Based on this analysis, mean current in binned phases was chosen as a key feature for the classification. To ensure other columns’ signals were not lost, the summary statistics (mean, max, min, std) of voltage, back\_emf and current, along with the position changes using the door leaf position. Position change was also used to infer the operation type (Open/Close), used later in classification as well. This is due to the differing distributions of current signals between the two operation types.   
*  A Random Forest Classifier is then run on the above stated features to produce the trained model.

Evaluation

* Separately, using the same segmentation and classification methods, a 80-20 train-validation split on the given train segments was completed to allow for us to validate the results. Validation showed that there is an IoU weighted F1 score of 1.00, suggesting significant separability between the signal values for normal and abnormal resistance cycles as well. 
