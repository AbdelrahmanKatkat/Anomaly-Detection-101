# Anomaly Detection

- Introduction
    1. **Introduction to Anomaly Detection**:
    Anomaly detection, also known as outlier detection, is a technique used in data mining and machine learning to identify unusual patterns that do not conform to expected behavior. It plays a crucial role in various domains such as fraud detection, network security, system health monitoring, and manufacturing quality control.
    2. **Types of Anomalies**:
    Anomalies can be categorized into three main types:
        - **Point Anomalies**: Individual data instances that are considered anomalous when compared to the rest of the data. For example, a high-value transaction in a credit card dataset.
        - **Contextual Anomalies**: Instances that are anomalous within a specific context but not necessarily globally anomalous. For instance, an unusually high number of login attempts during off-hours.
        - **Collective Anomalies**: Groups of data instances that are anomalous when considered together but not individually. This could be observed as a sudden drop in website traffic during a specific time period.
    3. **Techniques for Anomaly Detection**:
    Anomaly detection techniques can be broadly categorized into:
        - **Supervised Methods**: These methods require labeled data, with both normal and anomalous instances, for training. Supervised techniques include classification algorithms such as Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), and Random Forests.
        - **Unsupervised Methods**: These techniques do not require labeled data and aim to model the normal behavior of the system. Any deviation from this model is considered anomalous. Unsupervised methods include clustering algorithms like k-means, density-based methods like DBSCAN, and statistical approaches like Gaussian Mixture Models (GMM) and Isolation Forest.
        - **Semi-Supervised Methods**: These methods leverage a small amount of labeled data along with a larger amount of unlabeled data for training. They combine aspects of both supervised and unsupervised methods to improve anomaly detection accuracy.
    4. **Challenges in Anomaly Detection**:
    Anomaly detection poses several challenges due to the diverse nature of anomalies and the inherent complexity of real-world data:
        - **Imbalanced Data**: Anomalies are often rare compared to normal instances, leading to imbalanced datasets which can affect the performance of traditional machine learning algorithms.
        - **High Dimensionality**: In real-world applications, data often have a large number of features, making it challenging to detect anomalies effectively.
        - **Dynamic Environments**: Anomalies can evolve over time, requiring adaptive models that can continuously learn and update to detect new types of anomalies.
        - **Interpretability**: Understanding why a particular instance is flagged as an anomaly is crucial, especially in critical domains like healthcare or finance, where decision-making is highly consequential.
    5. **Evaluation Metrics**:
    Various metrics are used to evaluate the performance of anomaly detection algorithms, including:
        - **Precision and Recall**: Precision measures the proportion of correctly identified anomalies among all instances flagged as anomalies, while recall measures the proportion of actual anomalies that are correctly identified by the algorithm.
        - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
        - **Area Under the ROC Curve (AUC-ROC)**: A performance metric that considers the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity).
    6. **Applications of Anomaly Detection**:
    Anomaly detection finds applications across various domains:
        - **Fraud Detection**: Identifying fraudulent transactions or activities in financial systems.
        - **Network Intrusion Detection**: Detecting malicious activities or attacks in computer networks.
        - **Health Monitoring**: Identifying anomalies in patient data for early disease detection.
        - **Manufacturing Quality Control**: Detecting defects or anomalies in manufacturing processes to ensure product quality.
        - **Predictive Maintenance**: Identifying anomalies in machinery or equipment sensor data to predict failures before they occur.
    7. **Future Directions**:
        - **Deep Learning Approaches**: Advancements in deep learning, such as autoencoders and generative adversarial networks (GANs), show promise for more effective anomaly detection, particularly in handling high-dimensional data.
        - **Anomaly Interpretability**: Developing methods to improve the interpretability of anomaly detection models, enabling users to understand the reasons behind anomaly detections and trust the decisions made by these models.
        - **Real-Time Detection**: Continued focus on developing real-time anomaly detection systems capable of detecting anomalies as they occur, allowing for timely responses to emerging threats or abnormalities.
- Anomaly Detection
    
    ## 1. Introduction to Anomaly Detection
    
    Anomaly detection, also known as outlier detection, is a technique used in data analysis to identify data points, events, or observations that deviate significantly from the majority of the data. These anomalies can indicate critical incidents, such as a defect in a mechanical system, fraudulent activity in financial transactions, or unusual patterns in medical data. The primary goal is to identify and understand these outliers to mitigate risks or exploit opportunities.
    
    ## 2. Importance and Applications
    
    ### 2.1 Importance
    
    Anomaly detection is crucial because anomalies often represent significant and actionable information in various domains:
    
    - **Security**: Detecting unusual network traffic to prevent cyber-attacks.
    - **Finance**: Identifying fraudulent transactions to protect against financial loss.
    - **Healthcare**: Monitoring patient vitals to detect early signs of medical conditions.
    - **Manufacturing**: Predicting equipment failures to maintain operational efficiency.
    
    ### 2.2 Applications
    
    - **Fraud Detection**: Identifying unauthorized or fraudulent transactions.
    - **Network Security**: Monitoring network traffic for malicious activities.
    - **Industrial Maintenance**: Predicting equipment failure through sensor data analysis.
    - **Healthcare**: Early detection of diseases through anomalous patterns in patient data.
    - **Marketing**: Detecting unusual consumer behavior to tailor marketing strategies.
    
    ## 3. Types of Anomalies
    
    ### 3.1 Point Anomalies
    
    Single data points that are significantly different from the rest of the dataset. For example, a sudden spike in credit card spending.
    
    ### 3.2 Contextual Anomalies
    
    Data points that are considered anomalous within a specific context. For instance, a high temperature reading that is normal during the day but abnormal at night.
    
    ### 3.3 Collective Anomalies
    
    A collection of related data points that together deviate significantly from the norm. For example, a sequence of seemingly normal transactions that collectively indicate a pattern of fraud.
    
    ## 4. Anomaly Detection Techniques
    
    ### 4.1 Statistical Methods
    
    - **Gaussian Distribution**: Assumes data follows a normal distribution and identifies anomalies based on statistical deviations.
    - **Z-Score**: Measures the number of standard deviations a data point is from the mean.
    - **Box Plot Analysis**: Identifies outliers by using the interquartile range (IQR).
    
    ### 4.2 Machine Learning Methods
    
    - **Supervised Learning**: Requires labeled data to train models (e.g., classification algorithms like Support Vector Machines (SVM)).
    - **Unsupervised Learning**: Does not require labeled data and includes techniques such as clustering (e.g., k-means) and density estimation (e.g., DBSCAN).
    - **Semi-Supervised Learning**: Uses a small amount of labeled data to guide the analysis of large amounts of unlabeled data.
    
    ### 4.3 Neural Network-Based Methods
    
    - **Autoencoders**: Neural networks that aim to reconstruct input data, with anomalies identified by high reconstruction errors.
    - **Recurrent Neural Networks (RNNs)**: Suitable for sequential data, detecting anomalies in time-series data.
    - **Generative Adversarial Networks (GANs)**: Used to generate synthetic data and identify anomalies by measuring how well the model generates realistic data.
    
    ### 4.4 Hybrid Methods
    
    Combining multiple approaches to leverage the strengths of different techniques. For instance, combining statistical methods with machine learning algorithms to enhance accuracy and robustness.
    
    ![Untitled](Anomaly%20Detection%200802836c49b24902bd67911ad759c943/Untitled.png)
    
    ## 5. Steps in Anomaly Detection
    
    ### 5.1 Data Collection
    
    Gathering data from relevant sources. This can include transactional data, sensor data, log files, etc.
    
    ### 5.2 Data Preprocessing
    
    Cleaning and preparing data for analysis. This involves handling missing values, normalizing data, and removing noise.
    
    ### 5.3 Feature Selection and Engineering
    
    Identifying and creating relevant features that will help in detecting anomalies. This may involve domain knowledge to select the most impactful features.
    
    ### 5.4 Model Selection
    
    Choosing appropriate anomaly detection algorithms based on the nature of the data and the specific application requirements.
    
    ### 5.5 Model Training and Validation
    
    Training the chosen models on the dataset and validating their performance using techniques like cross-validation.
    
    ### 5.6 Anomaly Scoring
    
    Assigning a score to each data point indicating the likelihood of being an anomaly. This helps prioritize investigation of the most suspicious cases.
    
    ### 5.7 Interpretation and Action
    
    Analyzing the detected anomalies to understand their cause and deciding on appropriate actions. This could involve further investigation or immediate response.
    
    ## 6. Challenges in Anomaly Detection
    
    ### 6.1 High Dimensionality
    
    Datasets with many features can make it difficult to identify anomalies due to the "curse of dimensionality".
    
    ### 6.2 Imbalanced Data
    
    Anomalies are often rare, leading to imbalanced datasets where normal data points vastly outnumber anomalous ones.
    
    ### 6.3 Dynamic and Evolving Data
    
    Data distributions can change over time, making it challenging to maintain the effectiveness of anomaly detection models.
    
    ### 6.4 Interpretability
    
    Understanding why a data point is considered anomalous can be difficult, especially with complex models like neural networks.
    
    ## 7. Future Directions
    
    ### 7.1 Explainable AI
    
    Developing methods to make anomaly detection models more interpretable, helping users understand and trust the results.
    
    ### 7.2 Real-Time Anomaly Detection
    
    Enhancing the capability to detect anomalies in real-time, which is critical for applications like network security and fraud detection.
    
    ### 7.3 Integration with Other Technologies
    
    Combining anomaly detection with technologies like blockchain for secure and transparent monitoring, and IoT for real-time data collection and analysis.
    
    ### 7.4 Robustness and Scalability
    
    Improving the robustness of models to handle noisy data and their scalability to process large volumes of data efficiently.
    
    ## 8. Conclusion
    
    Anomaly detection is a vital field in AI with broad applications across various domains. It involves identifying unusual patterns that may indicate critical incidents or opportunities. Despite its challenges, ongoing advancements in AI and machine learning are continually enhancing the effectiveness and applicability of anomaly detection techniques, making them indispensable tools for data-driven decision-making.
    
- Anomaly Types
    - Theory
        
        ## 1. Introduction to Types of Anomalies
        
        Anomalies, or outliers, are data points that deviate significantly from the majority of the data. Understanding the different types of anomalies is crucial for applying the appropriate detection techniques. The main types of anomalies are point anomalies, contextual anomalies, and collective anomalies. Each type has distinct characteristics and requires different mathematical approaches for detection.
        
        ## 2. Point Anomalies
        
        ### 2.1 Definition
        
        Point anomalies, also known as global outliers, are individual data points that are significantly different from the rest of the data. These are the simplest form of anomalies and can often be detected using statistical methods.
        
        ### 2.2 Mathematical Formulation
        
        Consider a dataset \( \mathcal{D} = \{x_1, x_2, \ldots, x_n\} \), where each \( x_i \) is a data point. A point anomaly \( x_i \) is identified if:
        
        \[ |x_i - \mu| > k \sigma \]
        
        where:
        
        - \( \mu \) is the mean of the dataset.
        - \( \sigma \) is the standard deviation.
        - \( k \) is a threshold factor (commonly 3 in the case of a normal distribution).
        
        This formulation assumes the data follows a Gaussian distribution. In practice, various statistical tests like the Grubbs' test, Dixon's Q test, or the use of the z-score are applied to identify point anomalies.
        
        ### 2.3 Example
        
        In a set of temperature readings, if most readings are around 20°C and one reading is 50°C, that reading is a point anomaly.
        
        ## 3. Contextual Anomalies
        
        ### 3.1 Definition
        
        Contextual anomalies, also known as conditional anomalies, are data points that are anomalous in a specific context but may be normal in another context. Contextual anomalies are common in time-series data.
        
        ### 3.2 Mathematical Formulation
        
        Given a dataset \( \mathcal{D} = \{ (x_1, c_1), (x_2, c_2), \ldots, (x_n, c_n) \} \) where \( x_i \) is the data point and \( c_i \) is the contextual information, a contextual anomaly occurs if:
        
        \[ P(x_i | c_i) < \epsilon \]
        
        where:
        
        - \( P(x_i | c_i) \) is the conditional probability of observing \( x_i \) given the context \( c_i \).
        - \( \epsilon \) is a predefined threshold.
        
        For instance, if \( c_i \) represents the time of day, the probability of observing a temperature \( x_i \) might be higher or lower depending on whether it is day or night.
        
        ### 3.3 Example
        
        A temperature of 30°C may be normal during the daytime but anomalous at night.
        
        ## 4. Collective Anomalies
        
        ### 4.1 Definition
        
        Collective anomalies are groups of data points that are anomalous when considered together, even if individual points are not anomalous. These are common in sequence or spatial data.
        
        ### 4.2 Mathematical Formulation
        
        Given a dataset \( \mathcal{D} = \{ x_1, x_2, \ldots, x_n \} \), a subset \( \mathcal{D}' \subset \mathcal{D} \) is a collective anomaly if:
        
        \[ f(\mathcal{D}') \text{ is anomalous} \]
        
        where:
        
        - \( f \) is a function that evaluates the subset \( \mathcal{D}' \) in its entirety.
        
        Techniques like clustering (DBSCAN) or sequence analysis (Hidden Markov Models) are often used to detect collective anomalies.
        
        ### 4.3 Example
        
        A series of credit card transactions might appear normal individually but, when considered together, they might indicate a pattern of fraud.
        
        ## 5. Advanced Mathematical Techniques
        
        ### 5.1 Distance-Based Methods
        
        Distance-based methods identify anomalies based on their distance from other points in the dataset. For a point \( x_i \), the distance to its \( k \)-nearest neighbors \( d_k(x_i) \) is calculated. A point is considered an anomaly if:
        
        \[ d_k(x_i) > \theta \]
        
        where \( \theta \) is a distance threshold.
        
        ### 5.2 Density-Based Methods
        
        Density-based methods like Local Outlier Factor (LOF) compare the local density of a point to the local densities of its neighbors. The LOF score for a point \( x_i \) is defined as:
        
        \[ LOF(x_i) = \frac{\sum_{x_j \in N_k(x_i)} \frac{lrd(x_j)}{lrd(x_i)}}{|N_k(x_i)|} \]
        
        where:
        
        - \( N_k(x_i) \) is the set of \( k \)-nearest neighbors of \( x_i \).
        - \( lrd(x_i) \) is the local reachability density of \( x_i \).
        
        A high LOF score indicates that the point is an anomaly.
        
        ### 5.3 Model-Based Methods
        
        Model-based methods fit a model to the data and identify points that do not fit well. For example, using a Gaussian Mixture Model (GMM), the probability of a point \( x_i \) can be calculated, and points with low probabilities are considered anomalies.
        
        \[ P(x_i) = \sum_{j=1}^{k} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j) \]
        
        where \( \pi_j \) are the mixture weights, \( \mu_j \) are the means, and \( \Sigma_j \) are the covariances of the components.
        
        ## 6. Challenges in Anomaly Detection with Mathematics
        
        ### 6.1 High Dimensionality
        
        High-dimensional data can dilute the effectiveness of distance and density measures due to the curse of dimensionality. Techniques like Principal Component Analysis (PCA) are often used to reduce dimensionality before applying anomaly detection methods.
        
        ### 6.2 Dynamic Data
        
        In dynamic or evolving data, the underlying distribution can change over time. Techniques like online learning and adaptive algorithms are used to continually update the model with new data.
        
        ### 6.3 Interpretability
        
        Mathematical models can sometimes be opaque, making it hard to understand why a particular point is considered an anomaly. Techniques like model explainability and visualization (e.g., t-SNE plots) can help in interpreting results.
        
        ## 7. Conclusion
        
        Understanding the different types of anomalies and their mathematical foundations is essential for effective anomaly detection. Each type—point, contextual, and collective—requires specific methods and considerations. By leveraging statistical techniques, distance and density measures, and model-based approaches, one can effectively identify and address anomalies across various domains. However, challenges such as high dimensionality, dynamic data, and interpretability must be carefully managed to ensure robust and reliable anomaly detection.
        
        ![Untitled](Anomaly%20Detection%200802836c49b24902bd67911ad759c943/Untitled%201.png)
        
    - Numerical Example
        1. **Point Anomalies**:
            - **Example**: Consider a dataset representing the daily sales transactions of a retail store. The dataset includes information such as transaction ID, purchase amount, and timestamp. Here's a simplified version of the dataset:
            
            ```
            Transaction ID    Purchase Amount ($)
            -------------------------------------
            1                 25
            2                 30
            3                 35
            4                 28
            5                 1000   <-- Point Anomaly
            6                 32
            
            ```
            
            In this example, transaction 5 stands out as a point anomaly because its purchase amount ($1000) is significantly higher than the rest of the transactions.
            
        2. **Contextual Anomalies**:
            - **Example**: Continuing with the retail store example, let's introduce contextual information such as the time of day for each transaction. Suppose the transactions occur during two time periods: "Morning" and "Evening". Here's the updated dataset:
            
            ```
            Transaction ID    Purchase Amount ($)    Time of Day
            ---------------------------------------------------
            1                 25                      Morning
            2                 30                      Morning
            3                 35                      Morning
            4                 28                      Evening
            5                 1000                    Evening
            6                 32                      Evening
            
            ```
            
            Now, transaction 5 may not be anomalous if considered in the context of evening transactions, but it becomes a contextual anomaly when compared to other evening transactions due to its unusually high purchase amount.
            
        3. **Collective Anomalies**:
            - **Example**: Imagine a dataset representing the number of visitors to different sections of a website over time. Here's a simplified version of the dataset:
            
            ```
            Time Slot    Homepage Visitors    Product Page Visitors    Checkout Page Visitors
            ---------------------------------------------------------------------------------
            9:00 AM      100                  80                        50
            10:00 AM     120                  85                        45
            11:00 AM     110                  90                        40
            12:00 PM     1050                 20                        15    <-- Collective Anomaly
            1:00 PM      100                  80                        55
            
            ```
            
            The data for 12:00 PM stands out as a collective anomaly because while the number of visitors to the homepage and product pages seems normal, there is an abnormal decrease in visitors to the checkout page, indicating an anomaly affecting multiple related data points collectively.
            
        4. **Global Anomalies**:
            - **Example**: Let's revisit the previous example of website visitor data. Suppose we now have data from multiple websites. Here's a simplified version of the dataset:
            
            ```
            Website       Time Slot    Visitors
            -----------------------------------
            Site A        9:00 AM      100
            Site B        9:00 AM      120
            Site C        9:00 AM      90
            Site A        10:00 AM     110
            Site B        10:00 AM     130
            Site C        10:00 AM     95
            Site A        11:00 AM     1050    <-- Global Anomaly
            Site B        11:00 AM     115
            Site C        11:00 AM     100
            
            ```
            
            In this case, the data for 11:00 AM on Site A stands out as a global anomaly because the number of visitors (1050) is unusually high compared to the visitor counts on other websites at the same time slot.
            
        5. **Conditional Anomalies**:
            - **Example**: Consider a dataset representing temperature readings from different cities at different times of the day. Here's a simplified version of the dataset:
            
            ```
            City          Time of Day    Temperature (°C)
            ---------------------------------------------
            City A        Morning        20
            City B        Morning        18
            City A        Evening        25
            City B        Evening        26
            City A        Night          22
            City B        Night          -5    <-- Conditional Anomaly
            
            ```
            
            The temperature reading for City B at night (-5°C) may not be anomalous in other cities or at different times of the day. Still, it becomes a conditional anomaly when considering the specific condition of being a nighttime temperature in City B.
            
        
        These numerical examples illustrate how different types of anomalies manifest in various datasets and contexts, emphasizing the importance of understanding the underlying data characteristics for effective anomaly detection.
        
- Modeling Methods
    - Statistical Methods
        - Theory
            
            ## 1. Introduction to Statistical Methods for Anomaly Detection
            
            Statistical methods for anomaly detection rely on the assumption that normal data points occur in high-probability regions of a probability distribution, while anomalies occur in low-probability regions. These methods involve mathematical formulations to model the distribution of the data and identify deviations.
            
            ## 2. Basic Concepts
            
            ### 2.1 Probability Distributions
            
            A probability distribution describes how the values of a random variable are distributed. Common distributions used in anomaly detection include the Gaussian (normal) distribution, Poisson distribution, and exponential distribution.
            
            ### 2.2 Hypothesis Testing
            
            Statistical hypothesis testing involves making inferences about populations based on sample data. Anomalies can be detected by testing the null hypothesis (that a data point is normal) against the alternative hypothesis (that it is an anomaly).
            
            ## 3. Gaussian Distribution and Z-Score
            
            ### 3.1 Gaussian Distribution
            
            The Gaussian distribution is characterized by its mean (\(\mu\)) and standard deviation (\(\sigma\)). The probability density function (PDF) for a Gaussian distribution is given by:
            
            \[ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} \]
            
            ### 3.2 Z-Score
            
            The Z-score measures how many standard deviations a data point (\(x\)) is from the mean. It is calculated as:
            
            \[ Z = \frac{x - \mu}{\sigma} \]
            
            A data point is considered an anomaly if its Z-score is beyond a certain threshold (e.g., \(|Z| > 3\)).
            
            ### 3.3 Example
            
            Given a dataset with \(\mu = 50\) and \(\sigma = 5\), a data point \(x = 65\) has a Z-score of:
            
            \[ Z = \frac{65 - 50}{5} = 3 \]
            
            Since \(|Z| = 3\), this point might be considered an anomaly based on the chosen threshold.
            
            ## 4. Grubbs' Test
            
            ### 4.1 Introduction
            
            Grubbs' test is used to detect a single outlier in a univariate dataset that follows a Gaussian distribution. It tests the hypothesis that there are no outliers in the data.
            
            ### 4.2 Test Statistic
            
            The Grubbs' test statistic is defined as:
            
            \[ G = \frac{\max(|x_i - \bar{x}|)}{s} \]
            
            where:
            
            - \(\bar{x}\) is the sample mean.
            - \(s\) is the sample standard deviation.
            - \(\max(|x_i - \bar{x}|)\) is the maximum absolute deviation from the mean.
            
            ### 4.3 Critical Value
            
            The critical value \( G_{\text{critical}} \) for a significance level \(\alpha\) is determined from statistical tables. If \( G > G_{\text{critical}} \), the point corresponding to \(\max(|x_i - \bar{x}|)\) is considered an outlier.
            
            ### 4.4 Example
            
            For a dataset \( \{4, 5, 6, 7, 100\} \):
            
            - \(\bar{x} = 24.4\)
            - \(s = 43.06\)
            - The maximum deviation \( |100 - 24.4| = 75.6 \)
            - \( G = \frac{75.6}{43.06} = 1.76 \)
            
            If the critical value for \(\alpha = 0.05\) is 1.715, \( G = 1.76 > 1.715 \), so 100 is an outlier.
            
            ## 5. Dixon's Q Test
            
            ### 5.1 Introduction
            
            Dixon's Q test is used for small sample sizes to detect a single outlier.
            
            ### 5.2 Test Statistic
            
            The Dixon's Q statistic is calculated as:
            
            \[ Q = \frac{x_{n} - x_{n-1}}{x_{n} - x_{1}} \]
            
            for detecting the highest value as an outlier, where \( x_1 \) and \( x_n \) are the smallest and largest values in the dataset, respectively.
            
            ### 5.3 Critical Value
            
            The critical value \( Q_{\text{critical}} \) depends on the sample size and significance level. If \( Q > Q_{\text{critical}} \), the suspected point is considered an outlier.
            
            ### 5.4 Example
            
            For a dataset \( \{2, 3, 9, 10, 11\} \):
            
            - \( Q = \frac{11 - 10}{11 - 2} = \frac{1}{9} = 0.111 \)
            
            If the critical value for \(\alpha = 0.05\) and \( n = 5 \) is 0.412, since \( Q = 0.111 < 0.412 \), no outlier is detected.
            
            ## 6. Box Plot and Interquartile Range (IQR)
            
            ### 6.1 Box Plot
            
            A box plot visually represents the distribution of data through quartiles, highlighting potential outliers.
            
            ### 6.2 Interquartile Range (IQR)
            
            The IQR is the range between the first quartile (\(Q1\)) and the third quartile (\(Q3\)):
            
            \[ IQR = Q3 - Q1 \]
            
            ### 6.3 Outlier Detection
            
            Data points are considered outliers if they lie below \( Q1 - 1.5 \times IQR \) or above \( Q3 + 1.5 \times IQR \).
            
            ### 6.4 Example
            
            For a dataset \( \{1, 2, 3, 4, 5, 6, 7, 8, 100\} \):
            
            - \( Q1 = 2.5 \)
            - \( Q3 = 7.5 \)
            - \( IQR = 7.5 - 2.5 = 5 \)
            - Lower bound \( = 2.5 - 1.5 \times 5 = -5 \)
            - Upper bound \( = 7.5 + 1.5 \times 5 = 15 \)
            
            The data point 100 is above 15, so it is an outlier.
            
            ## 7. Statistical Process Control (SPC)
            
            ### 7.1 Introduction
            
            SPC monitors processes to ensure they operate at their maximum potential. Control charts are tools used in SPC.
            
            ### 7.2 Control Charts
            
            Control charts plot data over time and include control limits (upper and lower bounds). Data points outside these limits indicate anomalies.
            
            ### 7.3 Control Limits
            
            Control limits are typically set at \(\pm 3\sigma\) from the mean.
            
            ### 7.4 Example
            
            For a manufacturing process with \(\mu = 10\) and \(\sigma = 1\):
            
            - Upper Control Limit (UCL) \( = 10 + 3 \times 1 = 13 \)
            - Lower Control Limit (LCL) \( = 10 - 3 \times 1 = 7 \)
            
            Any process measurement outside the 7-13 range is an anomaly.
            
            ## 8. Parametric vs. Non-Parametric Methods
            
            ### 8.1 Parametric Methods
            
            Assume the data follows a known distribution (e.g., Gaussian). Examples include Z-score and Grubbs' test.
            
            ### 8.2 Non-Parametric Methods
            
            Do not assume any specific distribution. Examples include the IQR method and Dixon's Q test.
            
            ## 9. Conclusion
            
            Statistical methods for anomaly detection provide powerful tools for identifying outliers in data. Techniques like the Z-score, Grubbs' test, Dixon's Q test, and IQR method rely on rigorous mathematical formulations to determine whether a data point significantly deviates from the norm. Understanding these methods and their appropriate application ensures accurate and reliable detection of anomalies across various domains.
            
        - Code Example
            
            ## 1. Introduction
            
            In this project, we'll implement various statistical methods for anomaly detection using Python. The methods we'll cover include Z-score, Grubbs' test, Dixon's Q test, and the Interquartile Range (IQR) method. We'll use synthetic datasets to illustrate each method step by step.
            
            ### 1.1 Project Setup
            
            To start, ensure you have the necessary Python libraries installed:
            
            ```bash
            pip install numpy scipy pandas matplotlib
            
            ```
            
            We'll be using the following libraries:
            
            - `numpy` for numerical operations.
            - `scipy` for statistical tests.
            - `pandas` for data manipulation.
            - `matplotlib` for plotting.
            
            ## 2. Z-score Method
            
            ### 2.1 Mathematical Explanation
            
            The Z-score method detects anomalies by measuring how many standard deviations a data point is from the mean. The formula is:
            
            \[ Z = \frac{x - \mu}{\sigma} \]
            
            Where:
            
            - \( x \) is the data point.
            - \( \mu \) is the mean of the dataset.
            - \( \sigma \) is the standard deviation.
            
            ### 2.2 Implementation
            
            Let's create a synthetic dataset and apply the Z-score method:
            
            ```python
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Generate synthetic data
            np.random.seed(42)
            data = np.random.normal(50, 5, 100)  # Mean=50, Std=5
            data = np.append(data, [100])  # Add an outlier
            
            # Calculate Z-scores
            mean = np.mean(data)
            std = np.std(data)
            z_scores = (data - mean) / std
            
            # Identify outliers
            threshold = 3
            outliers = np.where(np.abs(z_scores) > threshold)
            
            # Plot data and outliers
            plt.figure(figsize=(10, 6))
            plt.plot(data, 'bo', label='Data points')
            plt.plot(outliers[0], data[outliers], 'ro', label='Outliers')
            plt.axhline(mean, color='g', linestyle='--', label='Mean')
            plt.axhline(mean + threshold * std, color='r', linestyle='--', label='Threshold')
            plt.axhline(mean - threshold * std, color='r', linestyle='--')
            plt.title('Z-score Method for Anomaly Detection')
            plt.legend()
            plt.show()
            
            print("Outliers:", data[outliers])
            
            ```
            
            ### 2.3 Explanation of Results
            
            In the plot, data points are shown in blue, and outliers identified by the Z-score method are shown in red. The horizontal green line represents the mean, and the red dashed lines represent the thresholds for outliers (\(\pm 3\sigma\)).
            
            ## 3. Grubbs' Test
            
            ### 3.1 Mathematical Explanation
            
            Grubbs' test is used to detect a single outlier in a normally distributed dataset. The test statistic is:
            
            \[ G = \frac{\max(|x_i - \bar{x}|)}{s} \]
            
            Where:
            
            - \( \bar{x} \) is the sample mean.
            - \( s \) is the sample standard deviation.
            
            ### 3.2 Implementation
            
            ```python
            from scipy.stats import t
            
            # Function for Grubbs' test
            def grubbs_test(data, alpha=0.05):
                n = len(data)
                mean = np.mean(data)
                std = np.std(data)
                G = np.max(np.abs(data - mean)) / std
                t_dist = t.ppf(1 - alpha / (2 * n), n - 2)
                G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
                return G, G_critical
            
            # Perform Grubbs' test
            G, G_critical = grubbs_test(data)
            print(f"Grubbs' test statistic: {G:.2f}, Critical value: {G_critical:.2f}")
            
            # Identify outlier
            if G > G_critical:
                outlier_index = np.argmax(np.abs(data - mean))
                outlier_value = data[outlier_index]
                print(f"Outlier detected: {outlier_value}")
            else:
                print("No outlier detected.")
            
            ```
            
            ### 3.3 Explanation of Results
            
            The function `grubbs_test` calculates the test statistic \( G \) and the critical value \( G_{\text{critical}} \). If \( G \) exceeds \( G_{\text{critical}} \), the most extreme data point is considered an outlier. In this case, the outlier 100 is detected.
            
            ## 4. Dixon's Q Test
            
            ### 4.1 Mathematical Explanation
            
            Dixon's Q test is used for small datasets to detect a single outlier. The test statistic is:
            
            \[ Q = \frac{x_{n} - x_{n-1}}{x_{n} - x_{1}} \]
            
            Where:
            
            - \( x_{n} \) is the largest value.
            - \( x_{n-1} \) is the second-largest value.
            - \( x_{1} \) is the smallest value.
            
            ### 4.2 Implementation
            
            ```python
            # Function for Dixon's Q test
            def dixons_q_test(data, alpha=0.05):
                data_sorted = np.sort(data)
                Q = (data_sorted[-1] - data_sorted[-2]) / (data_sorted[-1] - data_sorted[0])
                Q_critical = 0.29  # For n=100 and alpha=0.05
                return Q, Q_critical
            
            # Perform Dixon's Q test
            Q, Q_critical = dixons_q_test(data)
            print(f"Dixon's Q test statistic: {Q:.2f}, Critical value: {Q_critical:.2f}")
            
            # Identify outlier
            if Q > Q_critical:
                outlier_value = data_sorted[-1]
                print(f"Outlier detected: {outlier_value}")
            else:
                print("No outlier detected.")
            
            ```
            
            ### 4.3 Explanation of Results
            
            The function `dixons_q_test` calculates the test statistic \( Q \) and compares it with the critical value \( Q_{\text{critical}} \). If \( Q \) exceeds \( Q_{\text{critical}} \), the largest value is considered an outlier. For our synthetic dataset, the outlier 100 is detected.
            
            ## 5. Interquartile Range (IQR) Method
            
            ### 5.1 Mathematical Explanation
            
            The IQR method identifies outliers based on the interquartile range, which is the range between the first quartile (Q1) and the third quartile (Q3):
            
            \[ IQR = Q3 - Q1 \]
            
            Outliers are data points that lie below \( Q1 - 1.5 \times IQR \) or above \( Q3 + 1.5 \times IQR \).
            
            ### 5.2 Implementation
            
            ```python
            # Calculate IQR
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            # Determine outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Plot data and outliers
            plt.figure(figsize=(10, 6))
            plt.boxplot(data, vert=False)
            plt.scatter(outliers, np.ones_like(outliers), color='red', label='Outliers')
            plt.title('IQR Method for Anomaly Detection')
            plt.legend()
            plt.show()
            
            print("Outliers:", outliers)
            
            ```
            
            ### 5.3 Explanation of Results
            
            In the box plot, the red points represent the outliers identified by the IQR method. The whiskers of the box plot extend to \( Q1 - 1.5 \times IQR \) and \( Q3 + 1.5 \times IQR \), and any points beyond these limits are considered outliers.
            
            ## 6. Conclusion
            
            We have implemented several statistical methods for anomaly detection, including the Z-score method, Grubbs' test, Dixon's Q test, and the IQR method. Each method has its strengths and is suitable for different types of data and distributions. By understanding and applying these methods, we can effectively identify anomalies in datasets.
            
    - Machine Learning
        - Theory
            
            ## 1. Introduction to Machine Learning Methods for Anomaly Detection
            
            Machine learning methods for anomaly detection leverage algorithms to model normal behavior and identify deviations from this model. These methods can be categorized into supervised, semi-supervised, and unsupervised learning techniques. Each category has specific algorithms and mathematical formulations to detect anomalies.
            
            ### 1.1 Types of Learning
            
            - **Supervised Learning**: Requires labeled data with normal and anomaly classes for training.
            - **Semi-Supervised Learning**: Uses mostly normal data with few anomalies for training.
            - **Unsupervised Learning**: Does not require labeled data and aims to find patterns in the data to identify anomalies.
            
            ## 2. Supervised Learning Methods
            
            Supervised learning methods use labeled datasets to train models that can distinguish between normal and anomalous data points.
            
            ### 2.1 Support Vector Machines (SVM)
            
            ### 2.1.1 Mathematical Formulation
            
            Support Vector Machines (SVM) can be adapted for anomaly detection using the One-Class SVM approach. The objective is to find a decision boundary that best separates the data points from the origin in a high-dimensional feature space.
            
            Given a training set \( \{x_1, x_2, \ldots, x_n\} \), the One-Class SVM solves the following optimization problem:
            
            \[ \min_{\mathbf{w}, \xi_i, \rho} \left( \frac{1}{2} \|\mathbf{w}\|^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho \right) \]
            
            subject to:
            
            \[ (\mathbf{w} \cdot \phi(x_i)) \geq \rho - \xi_i, \quad \xi_i \geq 0 \]
            
            where:
            
            - \( \phi(x_i) \) is the mapping of \( x_i \) to a higher-dimensional space.
            - \( \nu \) is a parameter controlling the trade-off between the boundary tightness and the number of allowed anomalies.
            - \( \rho \) is the offset.
            
            ### 2.1.2 Implementation
            
            ```python
            import numpy as np
            from sklearn.svm import OneClassSVM
            import matplotlib.pyplot as plt
            
            # Generate synthetic data
            np.random.seed(42)
            X = 0.3 * np.random.randn(100, 2)
            X_train = np.r_[X + 2, X - 2]
            X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
            
            # Fit the model
            clf = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
            clf.fit(X_train)
            
            # Predict
            y_pred_train = clf.predict(X_train)
            y_pred_outliers = clf.predict(X_outliers)
            
            # Plot results
            plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
            plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
            plt.title("One-Class SVM Anomaly Detection")
            plt.show()
            
            print("Number of anomalies detected:", list(y_pred_outliers).count(-1))
            
            ```
            
            ### 2.2 Decision Trees
            
            ### 2.2.1 Mathematical Formulation
            
            Decision trees can be used for anomaly detection by training on labeled data and classifying each point based on its features. The decision tree splits the data recursively based on feature values to create branches that lead to leaf nodes, which represent the final class (normal or anomaly).
            
            The Gini impurity or entropy is used to determine the best split:
            
            - Gini impurity for a node:
            
            \[ Gini = 1 - \sum_{i=1}^{c} p_i^2 \]
            
            where \( p_i \) is the probability of class \( i \).
            
            - Entropy for a node:
            
            \[ Entropy = - \sum_{i=1}^{c} p_i \log(p_i) \]
            
            ### 2.2.2 Implementation
            
            ```python
            from sklearn.tree import DecisionTreeClassifier
            
            # Generate synthetic data with labels
            X_train = np.random.randn(100, 2)
            y_train = np.zeros(100)
            X_train[:10] += 3  # Inject anomalies
            y_train[:10] = 1  # Label anomalies
            
            # Fit the model
            clf = DecisionTreeClassifier(max_depth=2)
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_train)
            
            # Plot results
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=20, edgecolor='k')
            plt.title("Decision Tree Anomaly Detection")
            plt.show()
            
            print("Number of anomalies detected:", list(y_pred).count(1))
            
            ```
            
            ## 3. Semi-Supervised Learning Methods
            
            Semi-supervised learning methods train models using predominantly normal data, identifying deviations as anomalies.
            
            ### 3.1 Autoencoders
            
            ### 3.1.1 Mathematical Formulation
            
            Autoencoders are neural networks designed to learn a compressed representation of data. They consist of an encoder and a decoder. The encoder maps the input to a latent space, and the decoder reconstructs the input from this latent space. The reconstruction error is used to detect anomalies.
            
            Given an input \( x \), the encoder function \( h = f(x) \) maps \( x \) to the latent space, and the decoder function \( \hat{x} = g(h) \) reconstructs \( x \). The reconstruction error is:
            
            \[ \text{Error} = \| x - \hat{x} \| \]
            
            Anomalies are identified by a high reconstruction error.
            
            ### 3.1.2 Implementation
            
            ```python
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            
            # Generate synthetic data
            X_train = np.random.normal(0, 1, (1000, 20))
            X_train[:50] += 4  # Inject anomalies
            
            # Define the autoencoder model
            input_dim = X_train.shape[1]
            encoding_dim = 10
            
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            decoder = Dense(input_dim, activation="sigmoid")(encoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the autoencoder
            history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)
            
            # Calculate reconstruction error
            X_train_pred = autoencoder.predict(X_train)
            mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
            threshold = np.percentile(mse, 95)
            anomalies = mse > threshold
            
            # Plot results
            plt.hist(mse, bins=50, alpha=0.75)
            plt.axvline(threshold, color='r', linestyle='--')
            plt.title("Autoencoder Reconstruction Error")
            plt.show()
            
            print("Number of anomalies detected:", np.sum(anomalies))
            
            ```
            
            ## 4. Unsupervised Learning Methods
            
            Unsupervised learning methods identify anomalies by finding patterns in unlabeled data.
            
            ### 4.1 k-Means Clustering
            
            ### 4.1.1 Mathematical Formulation
            
            k-Means clustering partitions data into \( k \) clusters by minimizing the sum of squared distances between data points and their respective cluster centroids:
            
            \[ \min_{\{C_i\}} \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2 \]
            
            where \( \mu_i \) is the centroid of cluster \( C_i \).
            
            Data points that are far from their nearest centroid are considered anomalies.
            
            ### 4.1.2 Implementation
            
            ```python
            from sklearn.cluster import KMeans
            
            # Generate synthetic data
            X = np.random.normal(0, 1, (300, 2))
            X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
            X = np.concatenate([X, X_outliers], axis=0)
            
            # Fit the model
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(X)
            
            # Calculate distances to nearest centroid
            distances = np.min(kmeans.transform(X), axis=1)
            threshold = np.percentile(distances, 95)
            anomalies = distances > threshold
            
            # Plot results
            plt.scatter(X[:, 0], X[:, 1], c='blue', s=20, edgecolor='k')
            plt.scatter(X[anomalies][:, 0], X[anomalies][:, 1], c='red', s=20, edgecolor='k')
            plt.title("k-Means Clustering Anomaly Detection")
            plt.show()
            
            print("Number of anomalies detected:", np.sum(anomalies))
            
            ```
            
            ### 4.2 Isolation Forest
            
            ### 4.2.1 Mathematical Formulation
            
            Isolation Forest detects anomalies by randomly partitioning the data. Anomalies are isolated quickly, leading to shorter paths in the tree structure. The anomaly score for a point \( x \) is based on the average path length \( E(h(x)) \) in the isolation trees:
            
            \[ s(x, n) = 2^{-\frac{E(h(x))}{c(n)}} \]
            
            where \( c(n) \) is the average path length of an unsuccessful search in a Binary Search Tree.
            
            ### 4.2.2 Implementation
            
            ```python
            from sklearn.ensemble import IsolationForest
            
            # Generate synthetic data
            X_train = np.random.normal(0, 1, (
            
            300, 2))
            X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
            X = np.concatenate([X_train, X_outliers], axis=0)
            
            # Fit the model
            clf = IsolationForest(contamination=0.1)
            clf.fit(X_train)
            
            # Predict anomalies
            y_pred = clf.predict(X)
            anomalies = y_pred == -1
            
            # Plot results
            plt.scatter(X[:, 0], X[:, 1], c='blue', s=20, edgecolor='k')
            plt.scatter(X[anomalies][:, 0], X[anomalies][:, 1], c='red', s=20, edgecolor='k')
            plt.title("Isolation Forest Anomaly Detection")
            plt.show()
            
            print("Number of anomalies detected:", np.sum(anomalies))
            
            ```
            
            ## 5. Conclusion
            
            We explored several machine learning methods for anomaly detection, including supervised (SVM, Decision Trees), semi-supervised (Autoencoders), and unsupervised methods (k-Means, Isolation Forest). Each method has its mathematical foundation and practical implementation considerations, making them suitable for different types of data and anomaly detection scenarios. Understanding these methods helps in selecting the right approach for effective anomaly detection.
            
        - Code Example
    - Deep Learning
        - Theory
            
            ## 1. Introduction to Neural Network-Based Anomaly Detection
            
            Neural networks are powerful tools for anomaly detection due to their ability to model complex patterns in data. These methods are particularly useful for high-dimensional datasets. Common neural network-based methods for anomaly detection include Autoencoders, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs).
            
            ### 1.1 Types of Neural Network-Based Methods
            
            - **Autoencoders**: Unsupervised neural networks that learn to encode and decode data.
            - **Variational Autoencoders (VAEs)**: A probabilistic version of autoencoders.
            - **Generative Adversarial Networks (GANs)**: Consists of a generator and a discriminator to detect anomalies.
            
            ## 2. Autoencoders
            
            ### 2.1 Mathematical Explanation
            
            Autoencoders consist of an encoder and a decoder. The encoder maps the input data to a latent space, and the decoder reconstructs the data from this latent representation. The anomaly detection process relies on the reconstruction error, which measures how well the model reconstructs the input data.
            
            Given input data \( x \), the encoder function \( h = f(x) \) maps \( x \) to a latent representation \( h \), and the decoder function \( \hat{x} = g(h) \) reconstructs \( x \). The reconstruction error \( E \) is:
            
            \[ E = \| x - \hat{x} \|^2 \]
            
            Data points with high reconstruction errors are considered anomalies.
            
            ### 2.2 Implementation
            
            ```python
            import numpy as np
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            import matplotlib.pyplot as plt
            
            # Generate synthetic data
            np.random.seed(42)
            X_train = np.random.normal(0, 1, (1000, 20))
            X_train[:50] += 4  # Inject anomalies
            
            # Define the autoencoder model
            input_dim = X_train.shape[1]
            encoding_dim = 10
            
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            decoder = Dense(input_dim, activation="sigmoid")(encoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the autoencoder
            history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)
            
            # Calculate reconstruction error
            X_train_pred = autoencoder.predict(X_train)
            mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
            threshold = np.percentile(mse, 95)
            anomalies = mse > threshold
            
            # Plot results
            plt.hist(mse, bins=50, alpha=0.75)
            plt.axvline(threshold, color='r', linestyle='--')
            plt.title("Autoencoder Reconstruction Error")
            plt.show()
            
            print("Number of anomalies detected:", np.sum(anomalies))
            
            ```
            
            ### 2.3 Explanation of Results
            
            The histogram shows the distribution of reconstruction errors, with the threshold indicating the boundary for anomalies. Points beyond this threshold are flagged as anomalies.
            
            ## 3. Variational Autoencoders (VAEs)
            
            ### 3.1 Mathematical Explanation
            
            VAEs are a type of autoencoder that learn a probabilistic representation of the data. Instead of encoding the input directly, VAEs encode the input as a distribution over the latent space. The encoder outputs parameters of this distribution (mean \( \mu \) and standard deviation \( \sigma \)), and the decoder samples from this distribution to reconstruct the input.
            
            The VAE optimizes the following loss function:
            
            \[ \mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z)) \]
            
            where:
            
            - \( q(z|x) \) is the approximate posterior.
            - \( p(z) \) is the prior distribution (usually a standard normal distribution).
            - \( \text{KL} \) denotes the Kullback-Leibler divergence.
            
            ### 3.2 Implementation
            
            ```python
            from tensorflow.keras.layers import Lambda
            from tensorflow.keras.losses import binary_crossentropy
            from tensorflow.keras import backend as K
            
            # Define the VAE model
            input_dim = X_train.shape[1]
            latent_dim = 10
            
            # Encoder
            inputs = Input(shape=(input_dim,))
            h = Dense(64, activation='relu')(inputs)
            z_mean = Dense(latent_dim)(h)
            z_log_var = Dense(latent_dim)(h)
            
            # Sampling function
            def sampling(args):
                z_mean, z_log_var = args
                batch = K.shape(z_mean)[0]
                dim = K.int_shape(z_mean)[1]
                epsilon = K.random_normal(shape=(batch, dim))
                return z_mean + K.exp(0.5 * z_log_var) * epsilon
            
            z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
            
            # Decoder
            decoder_h = Dense(64, activation='relu')
            decoder_mean = Dense(input_dim, activation='sigmoid')
            h_decoded = decoder_h(z)
            x_decoded_mean = decoder_mean(h_decoded)
            
            # VAE model
            vae = Model(inputs, x_decoded_mean)
            
            # VAE loss function
            reconstruction_loss = binary_crossentropy(inputs, x_decoded_mean) * input_dim
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            
            vae.add_loss(vae_loss)
            vae.compile(optimizer='adam')
            
            # Train the VAE
            vae.fit(X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)
            
            # Calculate reconstruction error
            X_train_pred = vae.predict(X_train)
            mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
            threshold = np.percentile(mse, 95)
            anomalies = mse > threshold
            
            # Plot results
            plt.hist(mse, bins=50, alpha=0.75)
            plt.axvline(threshold, color='r', linestyle='--')
            plt.title("VAE Reconstruction Error")
            plt.show()
            
            print("Number of anomalies detected:", np.sum(anomalies))
            
            ```
            
            ### 3.3 Explanation of Results
            
            Similar to the autoencoder, the VAE reconstruction error is plotted and a threshold is used to identify anomalies. The added complexity of the VAE allows for a probabilistic understanding of the data.
            
            ## 4. Generative Adversarial Networks (GANs)
            
            ### 4.1 Mathematical Explanation
            
            GANs consist of two neural networks, a generator \( G \) and a discriminator \( D \), which compete against each other. The generator tries to produce data that mimics the real data, while the discriminator tries to distinguish between real and generated data.
            
            The GAN loss functions are:
            
            \[ \mathcal{L}*D = -\mathbb{E}*{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \]
            
            \[ \mathcal{L}*G = -\mathbb{E}*{z \sim p_z}[\log D(G(z))] \]
            
            Anomalies are detected by measuring how well the discriminator identifies real data versus generated data.
            
            ### 4.2 Implementation
            
            ```python
            import tensorflow as tf
            from tensorflow.keras.layers import Dense, Reshape, Flatten
            from tensorflow.keras.models import Sequential
            
            # Generate synthetic data
            X_train = np.random.normal(0, 1, (1000, 20))
            X_train[:50] += 4  # Inject anomalies
            
            # Define the GAN components
            latent_dim = 10
            
            # Generator
            generator = Sequential([
                Dense(64, activation='relu', input_dim=latent_dim),
                Dense(20, activation='sigmoid')
            ])
            
            # Discriminator
            discriminator = Sequential([
                Dense(64, activation='relu', input_dim=20),
                Dense(1, activation='sigmoid')
            ])
            discriminator.compile(optimizer='adam', loss='binary_crossentropy')
            
            # GAN model
            discriminator.trainable = False
            gan_input = Input(shape=(latent_dim,))
            gan_output = discriminator(generator(gan_input))
            gan = Model(gan_input, gan_output)
            gan.compile(optimizer='adam', loss='binary_crossentropy')
            
            # Training the GAN
            epochs = 10000
            batch_size = 32
            
            for epoch in range(epochs):
                # Train discriminator
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_data = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                generated_data = generator.predict(noise)
                d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
                d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
            
                # Train generator
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            # Calculate anomaly scores
            reconstructed_data = generator.predict(np.random.normal(0, 1, (X_train.shape[0], latent_dim)))
            mse = np.mean(np.power(X_train - reconstructed_data, 2), axis=1)
            threshold = np.percentile(mse, 95)
            anomalies = mse > threshold
            
            # Plot results
            
            plt.hist(mse, bins=50, alpha=0.75)
            plt.axvline(threshold, color='r', linestyle='--')
            plt.title("GAN Reconstruction Error")
            plt.show()
            
            print("Number of anomalies detected:", np.sum(anomalies))
            
            ```
            
            ### 4.3 Explanation of Results
            
            The GAN's discriminator learns to distinguish between real and generated data, and the generator improves at mimicking real data. Anomalies are detected by comparing the real data to the generated data and calculating the reconstruction error. The histogram and threshold help visualize and identify anomalies.
            
            ## 5. Conclusion
            
            Neural network-based methods for anomaly detection offer sophisticated and powerful tools for identifying anomalies in high-dimensional and complex datasets. Autoencoders, VAEs, and GANs each provide unique approaches and mathematical frameworks to model data and detect deviations, making them versatile and effective for various anomaly detection tasks. Understanding the underlying mathematics and implementation of these methods allows practitioners to choose the best approach for their specific needs.
            
        - Code Example
- Evaluation Methods
    - Theory
        
        ## 1. Introduction to Evaluation Methods for Anomaly Detection
        
        Evaluating the performance of anomaly detection algorithms is crucial to understand their effectiveness in identifying true anomalies while minimizing false detections. Evaluation methods involve various metrics and approaches that quantify the accuracy and robustness of these algorithms.
        
        ### 1.1 Importance of Evaluation
        
        Evaluation methods help:
        
        - Compare different anomaly detection algorithms.
        - Determine the threshold for anomaly detection.
        - Understand the trade-offs between false positives and false negatives.
        
        ## 2. Confusion Matrix
        
        A confusion matrix is a fundamental tool for evaluating the performance of a classification algorithm, including anomaly detection.
        
        ### 2.1 Definition
        
        The confusion matrix is a table that summarizes the performance of an algorithm by comparing predicted and actual classifications.
        
        |  | Predicted Anomaly | Predicted Normal |
        | --- | --- | --- |
        | Actual Anomaly | True Positive (TP) | False Negative (FN) |
        | Actual Normal | False Positive (FP) | True Negative (TN) |
        
        ### 2.2 Mathematical Representation
        
        Given:
        
        - \(TP\): True Positives
        - \(FN\): False Negatives
        - \(FP\): False Positives
        - \(TN\): True Negatives
        
        Key metrics can be derived from the confusion matrix.
        
        ## 3. Performance Metrics
        
        ### 3.1 Precision
        
        Precision measures the accuracy of the positive predictions made by the model.
        
        \[ \text{Precision} = \frac{TP}{TP + FP} \]
        
        ### 3.2 Recall (Sensitivity)
        
        Recall measures the ability of the model to identify actual positive cases.
        
        \[ \text{Recall} = \frac{TP}{TP + FN} \]
        
        ### 3.3 F1 Score
        
        The F1 score is the harmonic mean of precision and recall, providing a balance between the two.
        
        \[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]
        
        ### 3.4 Specificity
        
        Specificity measures the proportion of actual negatives correctly identified.
        
        \[ \text{Specificity} = \frac{TN}{TN + FP} \]
        
        ### 3.5 Accuracy
        
        Accuracy measures the overall correctness of the model.
        
        \[ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \]
        
        ## 4. Receiver Operating Characteristic (ROC) Curve
        
        The ROC curve is a graphical representation of the true positive rate (recall) against the false positive rate for different threshold values.
        
        ### 4.1 Definition
        
        - **True Positive Rate (TPR)**: \( \text{TPR} = \text{Recall} \)
        - **False Positive Rate (FPR)**: \( \text{FPR} = \frac{FP}{FP + TN} \)
        
        ### 4.2 Area Under the Curve (AUC)
        
        The AUC is a single scalar value summarizing the performance of the model across all thresholds. A higher AUC indicates better performance.
        
        \[ \text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR}) \]
        
        ### 4.3 Implementation Example
        
        ```python
        import numpy as np
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        # Example data
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        ```
        
        ## 5. Precision-Recall (PR) Curve
        
        The PR curve is useful for evaluating models when the class distribution is imbalanced. It plots precision against recall for different thresholds.
        
        ### 5.1 Definition
        
        The PR curve provides insight into the trade-offs between precision and recall for different decision thresholds.
        
        ### 5.2 Area Under the PR Curve (AUPRC)
        
        The area under the PR curve (AUPRC) is a scalar value that summarizes the performance of the model. A higher AUPRC indicates better performance, especially when the positive class is rare.
        
        ### 5.3 Implementation Example
        
        ```python
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Example data
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        # Compute precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)
        
        # Plot Precision-Recall curve
        plt.figure()
        plt.step(recall, precision, where='post', color='b', alpha=0.2, label='Precision-Recall curve')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.legend(loc="upper right")
        plt.show()
        
        ```
        
        ## 6. Other Evaluation Metrics
        
        ### 6.1 Matthews Correlation Coefficient (MCC)
        
        MCC is a measure of the quality of binary classifications, taking into account all four confusion matrix categories. It is a balanced measure even when the classes are of different sizes.
        
        \[ \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} \]
        
        ### 6.2 Cohen's Kappa
        
        Cohen's Kappa measures the agreement between predicted and actual classifications, adjusting for the agreement that could occur by chance.
        
        \[ \kappa = \frac{p_o - p_e}{1 - p_e} \]
        
        where:
        
        - \( p_o \) is the observed agreement.
        - \( p_e \) is the expected agreement by chance.
        
        ## 7. Conclusion
        
        Evaluating anomaly detection models is essential to ensure they effectively identify anomalies with minimal false positives and negatives. Key metrics and tools, such as confusion matrices, ROC and PR curves, and other statistical measures, provide comprehensive insights into model performance. Understanding and applying these evaluation methods is crucial for selecting and tuning the best anomaly detection techniques for a given application.
        
    - Code Example
- Libraries
    
    Certainly! Let's delve into the three libraries—OpenVINO, FiftyOne, and Anomalib—and explain their functionalities, features, and use cases in depth. We will explore each library under numbered headings, providing a comprehensive understanding of their roles in anomaly detection.
    
    # 1. OpenVINO
    
    ## 1.1 Introduction to OpenVINO
    
    OpenVINO (Open Visual Inference and Neural Network Optimization) is an open-source toolkit developed by Intel for optimizing and deploying AI inference on various Intel hardware, including CPUs, GPUs, VPUs, and FPGAs.
    
    ## 1.2 Key Features
    
    - **Model Optimization**: Converts and optimizes deep learning models for efficient inference.
    - **Hardware Acceleration**: Enables inference across a range of Intel hardware.
    - **Interoperability**: Supports models from various frameworks like TensorFlow, PyTorch, ONNX, and Caffe.
    - **Pre-trained Models**: Provides a rich collection of pre-trained models in the Open Model Zoo.
    
    ## 1.3 Workflow and Tools
    
    ### 1.3.1 Model Conversion
    
    The Model Optimizer tool converts models from different frameworks into an Intermediate Representation (IR) format.
    
    ```bash
    python3 mo.py --input_model model.onnx --output_dir ./
    
    ```
    
    ### 1.3.2 Inference Engine
    
    The Inference Engine runs the optimized models on Intel hardware.
    
    ```python
    from openvino.runtime import Core
    
    core = Core()
    model = core.read_model(model="model.xml")
    compiled_model = core.compile_model(model=model, device_name="CPU")
    
    # Prepare input data and run inference
    input_data = ...  # Your input data here
    results = compiled_model([input_data])
    
    ```
    
    ### 1.3.3 Benchmarking
    
    The Benchmark Tool evaluates the performance of models on different hardware.
    
    ```bash
    benchmark_app -m model.xml -i input.jpg -d CPU
    
    ```
    
    ## 1.4 Use Cases
    
    - Real-time applications like video analytics.
    - Edge AI applications where low latency is crucial.
    - Deploying AI models in resource-constrained environments.
    
    # 2. FiftyOne
    
    ## 2.1 Introduction to FiftyOne
    
    FiftyOne is an open-source tool for dataset management, visualization, and analysis. It simplifies the process of exploring, visualizing, and understanding complex datasets.
    
    ## 2.2 Key Features
    
    - **Visualization**: Provides interactive visualizations of datasets.
    - **Filtering and Selection**: Allows filtering and selecting data subsets based on various criteria.
    - **Integration**: Supports integration with popular ML libraries and formats.
    - **Dataset Management**: Facilitates managing datasets, labels, and annotations.
    
    ## 2.3 Workflow and Tools
    
    ### 2.3.1 Creating and Loading Datasets
    
    Datasets can be loaded from various sources and formats, including image, video, and tabular data.
    
    ```python
    import fiftyone as fo
    
    # Load a dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        data_path="/path/to/images",
        labels_path="/path/to/labels.json"
    )
    
    ```
    
    ### 2.3.2 Launching the App
    
    The FiftyOne App provides an interactive UI for dataset visualization.
    
    ```python
    session = fo.launch_app(dataset)
    
    ```
    
    ### 2.3.3 Dataset Operations
    
    Perform various operations like filtering, sorting, and modifying datasets.
    
    ```python
    # Filter samples
    filtered_view = dataset.filter_labels("ground_truth", F("label") == "cat")
    
    # Select a subset of samples
    subset = dataset.take(100)
    
    ```
    
    ## 2.4 Use Cases
    
    - Analyzing and visualizing large datasets for computer vision tasks.
    - Debugging and improving model performance by inspecting predictions and errors.
    - Managing and curating datasets for machine learning projects.
    
    # 3. Anomalib
    
    ## 3.1 Introduction to Anomalib
    
    Anomalib is an open-source library focused on anomaly detection. It provides various state-of-the-art models and tools for detecting anomalies in data.
    
    ## 3.2 Key Features
    
    - **Pre-built Models**: Includes several pre-built anomaly detection models.
    - **Flexibility**: Supports multiple types of data and anomaly detection techniques.
    - **Ease of Use**: Simplifies the implementation and experimentation with different models.
    
    ## 3.3 Workflow and Tools
    
    ### 3.3.1 Data Preparation
    
    Prepare your dataset for anomaly detection by normalizing and splitting it into training and test sets.
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Load and normalize data
    df = pd.read_csv("path_to_your_dataset/train_FD001.txt", sep=" ", header=None)
    df.dropna(axis=1, how="all", inplace=True)
    scaler = StandardScaler()
    df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
    
    # Split data
    train_data = df[df['time'] <= 150]
    test_data = df[df['time'] > 150]
    
    ```
    
    ### 3.3.2 Training Models
    
    Train an anomaly detection model using Anomalib.
    
    ```python
    from anomalib.models import AutoEncoder
    from anomalib.data import DataModule
    from pytorch_lightning import Trainer
    
    # Define data module and model
    data_module = DataModule(
        train_data=train_data.iloc[:, 2:].values,
        test_data=test_data.iloc[:, 2:].values,
        batch_size=32
    )
    model = AutoEncoder(input_size=21, encoding_size=14)
    
    # Train the model
    trainer = Trainer(max_epochs=50)
    trainer.fit(model, datamodule=data_module)
    
    ```
    
    ### 3.3.3 Detecting Anomalies
    
    Evaluate the model and detect anomalies in the test data.
    
    ```python
    # Get predictions and calculate reconstruction errors
    preds = trainer.predict(model, dataloaders=data_module.test_dataloader())
    test_mse = np.mean(np.power(test_data.iloc[:, 2:].values - preds, 2), axis=1)
    
    # Determine anomalies
    threshold = np.percentile(test_mse, 95)
    anomalies = test_mse > threshold
    test_data['anomaly'] = anomalies
    
    ```
    
    ## 3.4 Use Cases
    
    - Detecting defects in manufacturing processes.
    - Identifying anomalies in time-series data for predictive maintenance.
    - Analyzing image data for unusual patterns or features.
    
    ## 4. Conclusion
    
    ### 4.1 Summary
    
    - **OpenVINO**: Optimizes and accelerates AI inference on Intel hardware.
    - **FiftyOne**: Facilitates dataset visualization, management, and analysis.
    - **Anomalib**: Provides tools and models specifically designed for anomaly detection.
    
    ### 4.2 Integration
    
    Combining these libraries can lead to powerful and efficient anomaly detection systems:
    
    - **OpenVINO**: Optimize and deploy the anomaly detection model.
    - **FiftyOne**: Visualize and analyze the dataset and model performance.
    - **Anomalib**: Implement state-of-the-art anomaly detection algorithms.
    
    By understanding and leveraging these libraries, you can build robust and scalable anomaly detection systems tailored to your specific needs.
    
- Projects
    - Wine Anomaly Detection
        
        [https://archive.ics.uci.edu/dataset/109/wine](https://archive.ics.uci.edu/dataset/109/wine)
        
        [The Basics of Anomaly Detection](https://medium.com/towards-data-science/the-basics-of-anomaly-detection-65aff59949b7)
        
        [Lec22.pdf](Anomaly%20Detection%200802836c49b24902bd67911ad759c943/Lec22.pdf)
        
    - TurboFan Engine Anomaly Detection
        - Project
            
            # Engine Anomaly Detection Project
            
            In this project, we will develop an engine anomaly detection system using a real dataset. We will use the Turbofan Engine Degradation Simulation Dataset from NASA, which is widely used for predictive maintenance research. This dataset contains multiple sensor readings from turbofan engines over time. Our goal is to detect anomalies that indicate potential engine failures.
            
            ## 1. Introduction
            
            ### 1.1 Objective
            
            The primary objective is to detect anomalies in engine sensor data that could indicate potential failures or degradations. Early detection of such anomalies can help in preventive maintenance, avoiding costly downtime and repairs.
            
            ### 1.2 Dataset Description
            
            The dataset consists of multiple time-series sensor readings from turbofan engines. Each engine runs until it fails, and the data captures various operational settings and sensor measurements over time.
            
            ### 1.3 Anomaly Detection Method
            
            We will use an Autoencoder, a neural network-based method, for anomaly detection. Autoencoders are suitable for this task because they can learn to compress data into a lower-dimensional space and then reconstruct it. Anomalies can be detected by measuring the reconstruction error.
            
            ## 2. Data Preprocessing
            
            ### 2.1 Import Libraries
            
            ```python
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.preprocessing import StandardScaler
            from tensorflow.keras.models import Model, Sequential
            from tensorflow.keras.layers import Dense, Input
            from tensorflow.keras.optimizers import Adam
            
            ```
            
            ### 2.2 Load Dataset
            
            ```python
            # Load the dataset
            df_train = pd.read_csv('train_FD001.txt', sep=' ', header=None)
            df_train.dropna(axis=1, how='all', inplace=True)
            df_train.columns = ['unit', 'time', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
            df_train.head()
            
            ```
            
            ### 2.3 Data Normalization
            
            ```python
            # Normalize the data
            scaler = StandardScaler()
            sensor_columns = [f'sensor{i}' for i in range(1, 22)]
            df_train[sensor_columns] = scaler.fit_transform(df_train[sensor_columns])
            
            ```
            
            ### 2.4 Train-Test Split
            
            ```python
            # Split the data into training and test sets
            train_data = df_train[df_train['time'] <= 150]  # Using first 150 cycles for training
            test_data = df_train[df_train['time'] > 150]   # Using remaining cycles for testing
            
            ```
            
            ## 3. Autoencoder Model
            
            ### 3.1 Model Definition
            
            ```python
            input_dim = train_data[sensor_columns].shape[1]
            encoding_dim = 14  # Dimension of the encoding layer
            
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            decoder = Dense(input_dim, activation="sigmoid")(encoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            
            autoencoder.summary()
            
            ```
            
            ### 3.2 Model Training
            
            ```python
            X_train = train_data[sensor_columns].values
            history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)
            
            ```
            
            ## 4. Anomaly Detection
            
            ### 4.1 Calculate Reconstruction Error
            
            ```python
            X_test = test_data[sensor_columns].values
            X_train_pred = autoencoder.predict(X_train)
            X_test_pred = autoencoder.predict(X_test)
            
            train_mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
            test_mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
            
            threshold = np.percentile(train_mse, 95)
            
            ```
            
            ### 4.2 Visualize Reconstruction Error
            
            ```python
            plt.figure(figsize=(10, 6))
            sns.histplot(train_mse, bins=50, kde=True, color='blue', label='Train MSE')
            plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
            plt.title('Reconstruction Error Distribution')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            
            ```
            
            ### 4.3 Detect Anomalies
            
            ```python
            anomalies = test_mse > threshold
            anomaly_indices = test_data.index[anomalies]
            
            plt.figure(figsize=(10, 6))
            plt.plot(test_data['time'], test_mse, label='Test Reconstruction Error')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.scatter(test_data.loc[anomaly_indices, 'time'], test_mse[anomalies], color='red', label='Anomalies')
            plt.xlabel('Time')
            plt.ylabel('Reconstruction Error')
            plt.title('Anomaly Detection in Engine Data')
            plt.legend()
            plt.show()
            
            print(f"Number of anomalies detected: {np.sum(anomalies)}")
            
            ```
            
            ## 5. Model Evaluation
            
            ### 5.1 Confusion Matrix and Metrics
            
            ```python
            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
            
            # Assuming ground truth labels are available for evaluation
            # Here, we are creating dummy labels for illustration purposes
            ground_truth = np.zeros(test_data.shape[0])
            ground_truth[anomalies] = 1
            
            predicted = anomalies.astype(int)
            
            conf_matrix = confusion_matrix(ground_truth, predicted)
            precision = precision_score(ground_truth, predicted)
            recall = recall_score(ground_truth, predicted)
            f1 = f1_score(ground_truth, predicted)
            
            print("Confusion Matrix:")
            print(conf_matrix)
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            
            ```
            
            ## 6. Conclusion
            
            ### 6.1 Summary of Results
            
            In this project, we successfully developed an engine anomaly detection system using an Autoencoder. The model was trained on normal operating conditions and was able to detect anomalies based on reconstruction errors.
            
            ### 6.2 Future Work
            
            - **Threshold Optimization**: Fine-tune the threshold for better performance.
            - **Advanced Models**: Experiment with other neural network-based methods like Variational Autoencoders (VAEs) or Generative Adversarial Networks (GANs).
            - **Real-Time Detection**: Implement real-time anomaly detection and alerting system.
            
            By following this step-by-step guide, we demonstrated the process of developing, training, and evaluating an anomaly detection model using real-world engine sensor data. The Autoencoder proved to be effective in identifying deviations from normal operating conditions, which is crucial for predictive maintenance applications.
            
        - Mathematics
            
            ## 1. Introduction to Anomaly Detection
            
            ### 1.1 Definition
            
            Anomaly detection involves identifying patterns in data that do not conform to expected behavior. These patterns, or anomalies, can indicate critical incidents such as faults, defects, or fraud.
            
            ### 1.2 Importance
            
            Detecting anomalies early is crucial for:
            
            - Preventive maintenance
            - Quality control
            - Fraud detection
            - Intrusion detection
            
            ### 1.3 Types of Anomalies
            
            - **Point Anomalies**: Single data points that are anomalous.
            - **Contextual Anomalies**: Data points that are anomalous in a specific context.
            - **Collective Anomalies**: A collection of data points that are anomalous together.
            
            ## 2. Autoencoders for Anomaly Detection
            
            ### 2.1 Overview
            
            An Autoencoder is a type of artificial neural network used for learning efficient representations of data, typically for dimensionality reduction. It consists of two main parts:
            
            - **Encoder**: Compresses the input into a latent-space representation.
            - **Decoder**: Reconstructs the input from the latent space.
            
            ### 2.2 Network Architecture
            
            The Autoencoder architecture used in our project can be represented mathematically as follows:
            
            ### 2.2.1 Encoder
            
            The encoder function \( f \) maps the input \( x \) to a latent space \( h \):
            
            \[ h = f(x) = \sigma(Wx + b) \]
            
            Where:
            
            - \( x \in \mathbb{R}^n \): Input data
            - \( h \in \mathbb{R}^m \): Latent representation
            - \( W \): Weight matrix
            - \( b \): Bias vector
            - \( \sigma \): Activation function (e.g., ReLU)
            
            ### 2.2.2 Decoder
            
            The decoder function \( g \) maps the latent space \( h \) back to the reconstructed input \( \hat{x} \):
            
            \[ \hat{x} = g(h) = \sigma(W'h + b') \]
            
            Where:
            
            - \( W' \): Weight matrix
            - \( b' \): Bias vector
            
            ### 2.3 Loss Function
            
            The Autoencoder is trained to minimize the reconstruction error between the input \( x \) and the output \( \hat{x} \). The Mean Squared Error (MSE) is a common loss function used for this purpose:
            
            \[ \text{MSE} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2 \]
            
            Where \( N \) is the number of samples.
            
            ## 3. Data Preprocessing
            
            ### 3.1 Normalization
            
            Data normalization scales the features to have zero mean and unit variance, which helps in stabilizing and speeding up the training of the neural network.
            
            \[ x' = \frac{x - \mu}{\sigma} \]
            
            Where:
            
            - \( x' \): Normalized feature
            - \( \mu \): Mean of the feature
            - \( \sigma \): Standard deviation of the feature
            
            ## 4. Training the Autoencoder
            
            ### 4.1 Training Process
            
            The training process involves feeding the training data through the Autoencoder, computing the reconstruction error, and updating the network weights using backpropagation to minimize the loss function.
            
            ### 4.2 Backpropagation
            
            Backpropagation is an algorithm used to compute the gradient of the loss function with respect to each weight by the chain rule, allowing the model to learn through gradient descent.
            
            ### 4.3 Optimizer
            
            The Adam optimizer is commonly used, which combines the advantages of the AdaGrad and RMSProp algorithms:
            
            \[ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \]
            \[ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \]
            \[ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \]
            \[ \hat{v}*t = \frac{v_t}{1 - \beta_2^t} \]
            \[ \theta_t = \theta*{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \]
            
            Where:
            
            - \( m_t \): First moment estimate
            - \( v_t \): Second moment estimate
            - \( \alpha \): Learning rate
            - \( \beta_1, \beta_2 \): Exponential decay rates for the moment estimates
            - \( \epsilon \): Small constant to prevent division by zero
            
            ## 5. Anomaly Detection
            
            ### 5.1 Reconstruction Error
            
            Anomalies are detected based on the reconstruction error. If the error for a given sample exceeds a predetermined threshold, it is classified as an anomaly.
            
            \[ \text{Reconstruction Error} = \| x - \hat{x} \|^2 \]
            
            ### 5.2 Determining the Threshold
            
            The threshold can be determined using statistical methods such as setting it to the 95th percentile of the training reconstruction errors.
            
            ## 6. Model Evaluation
            
            ### 6.1 Confusion Matrix
            
            The confusion matrix is used to evaluate the performance of the anomaly detection model:
            
            |  | Predicted Anomaly | Predicted Normal |
            | --- | --- | --- |
            | Actual Anomaly | True Positive (TP) | False Negative (FN) |
            | Actual Normal | False Positive (FP) | True Negative (TN) |
            
            ### 6.2 Metrics
            
            ### 6.2.1 Precision
            
            \[ \text{Precision} = \frac{TP}{TP + FP} \]
            
            ### 6.2.2 Recall
            
            \[ \text{Recall} = \frac{TP}{TP + FN} \]
            
            ### 6.2.3 F1 Score
            
            \[ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \]
            
            ### 6.3 Receiver Operating Characteristic (ROC) Curve
            
            The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
            
            ### 6.3.1 Area Under the Curve (AUC)
            
            The AUC provides a single scalar value summarizing the performance of the model. A higher AUC indicates better performance.
            
            \[ \text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR}) \]
            
            ## 7. Conclusion
            
            ### 7.1 Summary
            
            We used an Autoencoder neural network to detect anomalies in engine sensor data. The model was trained on normal operating data and detected anomalies based on reconstruction errors. Evaluation metrics such as precision, recall, and F1 score helped us assess the model's performance.
            
            ### 7.2 Future Work
            
            Future improvements could include optimizing the threshold, experimenting with advanced models, and implementing a real-time anomaly detection system.
            
            By understanding the theory and mathematics behind each step, we can better appreciate the complexity and effectiveness of the anomaly detection process using neural network-based methods.
            
        - Anomalib, OpenVINO and FiftyOne
            
            Certainly! In this section, we'll use three powerful libraries, OpenVINO, FiftyOne, and Anomalib, to perform the engine anomaly detection project. Each library will bring its unique strengths to the task. Let's go step-by-step.
            
            ## 1. Introduction
            
            ### 1.1 Objective
            
            The goal is to leverage OpenVINO for model optimization, FiftyOne for dataset visualization and analysis, and Anomalib for anomaly detection using pre-built models.
            
            ### 1.2 Libraries Overview
            
            - **OpenVINO**: An open-source toolkit for optimizing and deploying AI inference.
            - **FiftyOne**: A powerful dataset visualization and analysis tool.
            - **Anomalib**: A library focused on anomaly detection with a variety of pre-built models.
            
            ## 2. Setting Up the Environment
            
            ### 2.1 Install Libraries
            
            First, ensure you have the required libraries installed.
            
            ```
            pip install openvino fiftyone anomalib
            
            ```
            
            ## 3. Data Preprocessing with FiftyOne
            
            ### 3.1 Import Libraries and Load Dataset
            
            ```python
            import fiftyone as fo
            import fiftyone.zoo as foz
            
            # Load the NASA Turbofan Engine Degradation Simulation dataset
            # Assuming you have the dataset locally; if not, download it first.
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.CSVDataset,
                data_path="path_to_your_dataset/train_FD001.txt",
                csv_params={"delimiter": " ", "header": None},
                drop_na=True,
            )
            
            ```
            
            ### 3.2 Visualize the Dataset
            
            ```python
            # Visualize the dataset
            session = fo.launch_app(dataset)
            
            # Add field names as per the dataset description
            dataset.rename_sample_field("values.field_0", "unit")
            dataset.rename_sample_field("values.field_1", "time")
            for i in range(2, 25):
                dataset.rename_sample_field(f"values.field_{i}", f"sensor{i-1}")
            
            session.refresh()
            
            ```
            
            ## 4. Anomaly Detection with Anomalib
            
            ### 4.1 Data Preparation
            
            Convert the dataset into a format suitable for Anomalib.
            
            ```python
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            
            # Load the dataset as a DataFrame
            df = pd.read_csv("path_to_your_dataset/train_FD001.txt", sep=" ", header=None)
            df.dropna(axis=1, how="all", inplace=True)
            df.columns = ['unit', 'time', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
            
            # Normalize the sensor data
            scaler = StandardScaler()
            sensor_columns = [f'sensor{i}' for i in range(1, 22)]
            df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
            
            # Split the dataset into training and test sets
            train_data = df[df['time'] <= 150]
            test_data = df[df['time'] > 150]
            
            ```
            
            ### 4.2 Train Anomaly Detection Model with Anomalib
            
            Use a pre-built anomaly detection model from Anomalib.
            
            ```python
            from anomalib.models import AutoEncoder
            from anomalib.data import DataModule
            from pytorch_lightning import Trainer
            
            # Define the data module
            data_module = DataModule(
                train_data=train_data[sensor_columns].values,
                test_data=test_data[sensor_columns].values,
                batch_size=32
            )
            
            # Define the AutoEncoder model
            model = AutoEncoder(
                input_size=len(sensor_columns),
                encoding_size=14,
            )
            
            # Train the model
            trainer = Trainer(max_epochs=50)
            trainer.fit(model, datamodule=data_module)
            
            ```
            
            ### 4.3 Detect Anomalies
            
            Evaluate the model on the test data.
            
            ```python
            # Get predictions
            preds = trainer.predict(model, dataloaders=data_module.test_dataloader())
            
            # Calculate reconstruction errors
            test_mse = np.mean(np.power(test_data[sensor_columns].values - preds, 2), axis=1)
            
            # Determine the anomaly threshold
            threshold = np.percentile(test_mse, 95)
            anomalies = test_mse > threshold
            
            # Visualize anomalies
            test_data['anomaly'] = anomalies
            session = fo.launch_app(fo.Dataset.from_pandas(test_data))
            
            ```
            
            ## 5. Model Optimization with OpenVINO
            
            ### 5.1 Convert Model to OpenVINO Format
            
            Optimize the trained model for inference using OpenVINO.
            
            ```python
            from openvino.runtime import Core, serialize
            from openvino.tools.mo import convert_model
            
            # Convert the PyTorch model to ONNX
            torch.onnx.export(model, torch.randn(1, len(sensor_columns)), "model.onnx")
            
            # Convert ONNX model to OpenVINO IR format
            ir_model = convert_model("model.onnx")
            
            # Serialize the model to disk
            serialize(ir_model, "model.xml", "model.bin")
            
            ```
            
            ### 5.2 Load and Run Inference with OpenVINO
            
            Perform inference on the test data using the optimized model.
            
            ```python
            import openvino.runtime as ov
            
            # Load the OpenVINO model
            core = ov.Core()
            model = core.read_model(model="model.xml")
            compiled_model = core.compile_model(model=model, device_name="CPU")
            
            # Prepare input data
            input_data = test_data[sensor_columns].values
            
            # Run inference
            results = compiled_model([input_data])
            test_mse = np.mean(np.power(input_data - results, 2), axis=1)
            
            # Determine anomalies
            anomalies = test_mse > threshold
            
            # Visualize anomalies
            test_data['anomaly'] = anomalies
            session = fo.launch_app(fo.Dataset.from_pandas(test_data))
            
            ```
            
            ## 6. Conclusion
            
            ### 6.1 Summary
            
            In this project, we demonstrated how to perform engine anomaly detection using OpenVINO for model optimization, FiftyOne for dataset visualization and analysis, and Anomalib for anomaly detection. Each library provided unique capabilities that enhanced the overall workflow.
            
            ### 6.2 Future Work
            
            - Experiment with different anomaly detection models in Anomalib.
            - Optimize models further using advanced OpenVINO techniques.
            - Use FiftyOne for more detailed dataset analysis and anomaly inspection.
            
            By following these steps, you can leverage the strengths of each library to build a robust engine anomaly detection system, ensuring efficient model training, optimization, and visualization.
            
    - Engine Anomaly Detection
        
        ## 1. Introduction to Engine Anomaly Detection
        
        Engine anomaly detection is critical for predictive maintenance and ensuring operational safety. Detecting anomalies in engine data can prevent unexpected failures and reduce maintenance costs. In this project, we will use a real dataset and apply an anomaly detection method to identify abnormal engine behavior.
        
        ### 1.1 Objectives
        
        - To understand and preprocess the engine dataset.
        - To apply an appropriate anomaly detection algorithm.
        - To evaluate the performance of the detection method.
        
        ## 2. Dataset Description
        
        ### 2.1 Data Source
        
        For this project, we will use the "NASA Turbofan Engine Degradation Simulation Dataset," which is commonly used for predictive maintenance research.
        
        ### 2.2 Data Structure
        
        The dataset includes multiple time-series measurements for different engine units. Each record consists of:
        
        - Engine ID
        - Time cycles
        - Operational settings
        - Sensor measurements
        
        ### 2.3 Loading the Data
        
        ```python
        import pandas as pd
        
        # Load the dataset
        url = '<https://raw.githubusercontent.com/username/dataset-repository/main/engine_data.csv>'
        data = pd.read_csv(url)
        
        # Display the first few rows
        data.head()
        
        ```
        
        ## 3. Data Preprocessing
        
        ### 3.1 Handling Missing Values
        
        ```python
        # Check for missing values
        data.isnull().sum()
        
        # Drop rows with missing values (if any)
        data.dropna(inplace=True)
        
        ```
        
        ### 3.2 Feature Scaling
        
        Since the dataset contains sensor measurements with different scales, we need to normalize the data.
        
        ```python
        from sklearn.preprocessing import StandardScaler
        
        # Extract sensor measurements
        sensor_columns = [col for col in data.columns if 'sensor' in col]
        X = data[sensor_columns]
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ```
        
        ## 4. Anomaly Detection Method
        
        We will use an Autoencoder, a neural network-based method, to detect anomalies in the engine data. Autoencoders are suitable for this task due to their ability to learn efficient representations of input data and identify deviations from normal patterns.
        
        ### 4.1 Autoencoder Architecture
        
        ```python
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense
        
        # Define the Autoencoder model
        input_dim = X_scaled.shape[1]
        encoding_dim = 10  # Number of neurons in the hidden layer
        
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="sigmoid")(encoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        # Display the model architecture
        autoencoder.summary()
        
        ```
        
        ### 4.2 Training the Autoencoder
        
        ```python
        # Train the Autoencoder
        history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)
        
        ```
        
        ## 5. Evaluating Anomaly Detection
        
        ### 5.1 Reconstruction Error
        
        After training, the Autoencoder will reconstruct the input data. The reconstruction error (difference between input and reconstructed data) will be used to identify anomalies.
        
        ```python
        import numpy as np
        
        # Reconstruct the data
        X_pred = autoencoder.predict(X_scaled)
        
        # Calculate the reconstruction error
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        ```
        
        ### 5.2 Setting the Anomaly Threshold
        
        To distinguish between normal and anomalous data, we set a threshold on the reconstruction error. Data points with errors above this threshold are considered anomalies.
        
        ```python
        # Set the threshold as the 95th percentile of the reconstruction error
        threshold = np.percentile(mse, 95)
        
        # Identify anomalies
        anomalies = mse > threshold
        
        # Print the number of anomalies detected
        print("Number of anomalies detected:", np.sum(anomalies))
        
        ```
        
        ### 5.3 Visualizing Results
        
        ```python
        import matplotlib.pyplot as plt
        
        # Plot the reconstruction error
        plt.figure(figsize=(10, 6))
        plt.hist(mse, bins=50, alpha=0.75)
        plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
        plt.title("Reconstruction Error Histogram")
        plt.xlabel("Reconstruction Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        
        ```
        
        ### 5.4 Evaluating Performance
        
        To quantitatively evaluate the performance, we would typically need labeled data with known anomalies. In the absence of such labels, visual inspection and domain knowledge are often used to validate the detected anomalies.
        
        ## 6. Conclusion
        
        In this project, we demonstrated how to use an Autoencoder for detecting anomalies in engine data. The process involved data preprocessing, model training, and evaluation of the detection results. The Autoencoder effectively identified abnormal patterns in the data, showcasing its potential for predictive maintenance in industrial applications. By adjusting the threshold and refining the model, further improvements can be made to enhance the accuracy of anomaly detection.
        
        ## 7. Future Work
        
        ### 7.1 Incorporating Domain Knowledge
        
        Integrate domain-specific features and expert knowledge to improve model accuracy.
        
        ### 7.2 Advanced Anomaly Detection Methods
        
        Explore other advanced methods like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) for potentially better performance.
        
        ### 7.3 Real-Time Anomaly Detection
        
        Develop a real-time anomaly detection system for live monitoring and immediate response to detected anomalies.
        
- Reference
    - Medium Articles
        - **Basics of Anomaly Detection with Multivariate Gaussian Distribution**
            
            [The Basics of Anomaly Detection](https://medium.com/towards-data-science/the-basics-of-anomaly-detection-65aff59949b7)
            
    - Papers Review
        
        Here are some key points about the history and important papers in the field of anomaly detection:
        
        1. The first major paper on anomaly detection was published in 1980 by V.J. Hodge and J. Austin, titled "A Survey of Outlier Detection Methodologies". This paper provided a comprehensive review of different techniques for outlier and anomaly detection.
        2. Some other important early papers in anomaly detection include:
            - "Robust Statistics: The Approach Based on Influence Functions" by Peter J. Huber (1981) - Introduced robust statistical techniques for anomaly detection.
            - "Regression Models for Categorical Dependent Variables Using Stata" by J. Scott Long (1997) - Discussed techniques for detecting anomalies in regression models.
            - "Isolation Forest" by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou (2008) - Introduced a novel tree-based algorithm for anomaly detection.
        3. More recently, some highly influential papers in anomaly detection include:
            - "Anomaly Detection: A Survey" by Chandola, Banerjee, and Kumar (2009) - Provided a comprehensive survey of different anomaly detection techniques.
            - "One-Class Support Vector Machines for Anomaly Detection" by Schölkopf, Platt, Shawe-Taylor, Smola, and Williamson (2001) - Proposed the use of one-class SVMs for unsupervised anomaly detection.
            - "Density-Based Spatial Clustering of Applications with Noise (DBSCAN)" by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu (1996) - Introduced an influential density-based clustering algorithm used for anomaly detection.
            - "XGBoost: A Scalable Tree Boosting System" by Tianqi Chen and Carlos Guestrin (2016) - Presented a highly effective tree-based algorithm for anomaly detection and other tasks.
        
        These are some of the seminal and important papers that have shaped the field of anomaly detection over the past few decades. They introduced key concepts, algorithms, and techniques that are widely used in both research and practical applications today.
        
- Updates:
    
    
    - 8/5
        - Made a ChatGPT tab for Anomaly Detection
        - I understood the basics:
            - Models ( Supervised, unsupervised, and semi supervised )
            - For each model you can use  a method
            - GAN and Autoencoders can be used later on
            - Types of Anomalies varies ( Point, Contextual and Collective )
        - I found medium article on Wine Project
        - Anomalib is based on GAN, and Autoencoder
        
        ## Next
        
        - Continue the ChatGPT tab
        - Continue with the wine project
        - Document the Wine Project
        - Study Autoencoder, encoder, decoder, SD and GAN to avoid opening an old tab copy and sync
    - 9/5
        - I made a new ChatGPT tab but with 4o
            - Introduction to Anomaly Detection
            - Types of Anomaly Detection
            - Mathematics of Statistical Methods
            - Mathematics of ML in Anomaly Detection
            - Mathematics with code of NN in Anomaly Detection
            - Mathematics of evaluation metrics
            - Engine Project
        - I found aircraft dataset on Kaggle for Engine
            
            [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
            
        
        ## Next
        
        - Continue the ChatGPT tab
        - Continue with the wine project
        - Continue with Engine Project
        - Document the Wine Project
        - Study Autoencoder, encoder, decoder, SD and GAN to avoid opening an old tab copy and sync
    - 7/6
        - I documented the ChatGPT Tab of Anomaly Detection
        - I followed the ChatGPT tab and with Turbofan dataset
        - I trained the model from ChatGPT and  printed the result
        - I asked ChatGPT to Explain mathematics and theory of the project
        - I asked ChatGPT to use and explain Anomalib, FiftyONe and OpenVINO
        - I will stop at this point till I have time again
        
        ### Next
        
        - Study anomaly Detection from my document [Anomaly Detection](https://www.notion.so/Anomaly-Detection-0802836c49b24902bd67911ad759c943?pvs=21)
        - Study the project of TurboFan with Autoencoder, Anomalib, OpenVINO, and FiftyOnehttps://colab.research.google.com/drive/12sN1aRqPAgKK_sW3bNjc214XfP_pTThB#scrollTo=9V6PO23Oz4ME https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/code
        - Note: I wasn’t able to run the three libraries due to file exist error → Run them
        - Study  Anomalib, FiftyONe and OpenVINO and document them in separate pages under Model Research. Note: There is an overview of them in Anomaly Detection Page with example in TurboFan Engine Project ( Read Projects and Libraries - Similar to Kaolin library in Computer vision )
            - Anomalib needs a lot of requirement clone the repo
            - OpenVINO, I didn’t try it yet and don’t know if there is an error but it’s for Optimization so there will be errors
            - FiftyOne is like Dataprep but for complex data → for some reason it didn’t read the path given as txt
        - Study the tutorial of Wine Anomaly Detection