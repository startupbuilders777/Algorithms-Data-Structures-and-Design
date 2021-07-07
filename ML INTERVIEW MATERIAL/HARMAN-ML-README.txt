1. 
    What is cross-validation? How to do it right?
    It’s a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. Mainly used in settings where the goal is prediction and one wants to estimate how accurately a model will perform in practice. The goal of cross-validation is to define a data set to test the model in the training phase (i.e. validation data set) in order to limit problems like overfitting, and get an insight on how the model will generalize to an independent data set.

    Examples: leave-one-out cross validation, K-fold cross validation

    How to do it right?

    the training and validation data sets have to be drawn from the same population
    predicting stock prices: trained for a certain 5-year period, it’s unrealistic to treat the subsequent 5-year a draw from the same population
    common mistake: for instance the step of choosing the kernel parameters of a SVM should be cross-validated as well
    Bias-variance trade-off for k-fold cross validation:

    Leave-one-out cross-validation: gives approximately unbiased estimates of the test error since each training set contains almost the entire data set (n−1 observations).

    But: we average the outputs of n fitted models, each of which is trained on an almost identical set of observations hence the outputs are highly correlated. Since the variance of a mean of quantities increases when correlation of these quantities increase, the test error estimate from a LOOCV has higher variance than the one obtained with k-fold cross validation

    Typically, we choose k=5 or k=10, as these values have been shown empirically to yield test error estimates that suffer neither from excessively high bias nor high variance.
























21) How to Use Label Smoothing for Regularization
    What is label smoothing and how to implement it in PyTorch
    Dimitris Poulopoulos
    Dimitris Poulopoulos
    Follow
    Mar 20 · 4 min read

    Image for post
    Photo by Dave on Unsplash
    Overfitting and probability calibration are two issues that arise when training deep learning models. There are a lot of regularization techniques in deep learning to address overfitting; weight decay, early stopping, dropout are some of the most popular ones. On the other hand, Platt’s scaling and isotonic regression are used for model calibration. But is there one method that fights both overfitting and over-confidence?
    Label smoothing is a regularization technique that perturbates the target variable, to make the model less certain of its predictions. It is viewed as a regularization technique because it restrains the largest logits fed into the softmax function from becoming much bigger than the rest. Moreover, the resulted model is better calibrated as a side-effect.
    In this story, we define label smoothing, implement a cross-entropy loss function that uses this technique and put it to the test. If you want to read more about model calibration please refer to the story below.
    Classifier calibration
    The why, when and how of model calibration for classification tasks
    towardsdatascience.com
    Label Smoothing
    Imagine that we have a multiclass classification problem. In such problems, the target variable is usually a one-hot vector, where we have 1 in the position of the correct class and 0s everywhere else.
    Label smoothing changes the target vector by a small amount ε. Thus, instead of asking our model to predict 1 for the right class, we ask it to predict 1-ε for the correct class and ε for all the others. So, the cross-entropy loss function with label smoothing is transformed into the formula below.
    Image for post
    In this formula, ce(x) denotes the standard cross-entropy loss of x (e.g. -log(p(x))), ε is a small positive number, i is the correct class and N is the number of classes.
    Intuitively, label smoothing restraints the logit value for the correct class to be closer to the logit values for other classes. In such way, it is used as a regularization technique and a method to fight model over-confidence.

18. Explain Eigenvalue and Eigenvector

    Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular linear transformation acts by flipping, compressing or stretching. Eigenvalue can be referred to as the strength of the transformation in the direction of eigenvector or the factor by which the compression occurs.

Q14: What’s the difference between a generative and discriminative model?
    Answer: A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

68) Give some situations where you will use an SVM over a RandomForest Machine Learning algorithm and vice-versa.

    SVM and Random Forest are both used in classification problems.

    a)      If you are sure that your data is outlier free and clean then go for SVM. It is the opposite -   if your data might contain outliers then Random forest would be the best choice

    b)      Generally, SVM consumes more computational power than Random Forest, so if you are constrained with memory go for Random Forest machine learning algorithm.

    c)  Random Forest gives you a very good idea of variable importance in your data, so if you want to have variable importance then choose Random Forest machine learning algorithm.

    d)      Random Forest machine learning algorithms are preferred for multiclass problems.

    e)     SVM is preferred in multi-dimensional problem set - like text classification

    but as a good data scientist, you should experiment with both of them and test for accuracy or rather you can use ensemble of many Machine Learning techniques.
77) 
    What is the advantage of performing dimensionality reduction before fitting an SVM?

    Support Vector Machine Learning Algorithm performs better in the reduced space. 
    It is beneficial to perform dimensionality reduction before fitting an SVM if the 
    number of features is large when compared to the number of observations.


22. 
    What’s the difference between Gaussian Mixture Model and K-Means?
    Let's says we are aiming to break them into three clusters. K-means will start with the assumption that a given data point belongs to one cluster.

    Choose a data point. At a given point in the algorithm, we are certain that a point belongs to a red cluster. In the next iteration, we might revise that belief, and be certain that it belongs to the green cluster. However, remember, in each iteration, we are absolutely certain as to which cluster the point belongs to. This is the "hard assignment".

    What if we are uncertain? What if we think, well, I can't be sure, but there is 70% chance it belongs to the red cluster, but also 10% chance its in green, 20% chance it might be blue. That's a soft assignment. The Mixture of Gaussian model helps us to express this uncertainty. It starts with some prior belief about how certain we are about each point's cluster assignments. As it goes on, it revises those beliefs. But it incorporates the degree of uncertainty we have about our assignment.

    Kmeans: find k to minimize (x−μk)^2

    Gaussian Mixture (EM clustering) : find k to minimize (x−μk)^2/σ^2

    The difference (mathematically) is the denominator “σ^2”, which means GM takes variance into consideration when it calculates the measurement. Kmeans only calculates conventional Euclidean distance. In other words, Kmeans calculate distance, while GM calculates “weighted” distance.

    K means:

    Hard assign a data point to one particular cluster on convergence.
    It makes use of the L2 norm when optimizing (Min {Theta} L2 norm point and its centroid coordinates).
    EM:

    Soft assigns a point to clusters (so it give a probability of any point belonging to any centroid).
    It doesn't depend on the L2 norm, but is based on the Expectation, i.e., the probability of the point belonging to a particular cluster. This makes K-means biased towards spherical clusters.



23. 
    Describe how Gradient Boosting works.
    The idea of boosting came out of the idea of whether a weak learner can be modified to become better.

    Gradient boosting relies on regression trees (even when solving a classification problem) which minimize MSE. Selecting a prediction for a leaf region is simple: to minimize MSE we should select an average target value over samples in the leaf. The tree is built greedily starting from the root: for each leaf a split is selected to minimize MSE for this step.

    To begin with, gradient boosting is an ensembling technique, which means that prediction is done by an ensemble of simpler estimators. While this theoretical framework makes it possible to create an ensemble of various estimators, in practice we almost always use GBDT — gradient boosting over decision trees.

    The aim of gradient boosting is to create (or "train") an ensemble of trees, given that we know how to train a single decision tree. This technique is called boosting because we expect an ensemble to work much better than a single estimator.

    Here comes the most interesting part. Gradient boosting builds an ensemble of trees one-by-one, then the predictions of the individual trees are summed: D(x)=d​tree 1​​(x)+d​tree 2​​(x)+...

    The next decision tree tries to cover the discrepancy between the target function f(x) and the current ensemble prediction by reconstructing the residual.

    For example, if an ensemble has 3 trees the prediction of that ensemble is: D(x)=d​tree 1​​(x)+d​tree 2​​(x)+d​tree 3​​(x). The next tree (tree 4) in the ensemble should complement well the existing trees and minimize the training error of the ensemble.

    In the ideal case we'd be happy to have: D(x)+d​tree 4​​(x)=f(x).

    To get a bit closer to the destination, we train a tree to reconstruct the difference between the target function and the current predictions of an ensemble, which is called the residual: R(x)=f(x)−D(x). Did you notice? If decision tree completely reconstructs R(x), the whole ensemble gives predictions without errors (after adding the newly-trained tree to the ensemble)! That said, in practice this never happens, so we instead continue the iterative process of ensemble building.

AdaBoost the First Boosting Algorithm
    The weak learners in AdaBoost are decision trees with a single split, called decision stumps for their shortness.

    AdaBoost works by weighting the observations, putting more weight on difficult to classify instances and less on those already handled well. New weak learners are added sequentially that focus their training on the more difficult patterns. Gradient boosting involves three elements:

    A loss function to be optimized.
    A weak learner to make predictions.
    An additive model to add weak learners to minimize the loss function.
    Loss Function
    The loss function used depends on the type of problem being solved. It must be differentiable, but many standard loss functions are supported and you can define your own. For example, regression may use a squared error and classification may use logarithmic loss. A benefit of the gradient boosting framework is that a new boosting algorithm does not have to be derived for each loss function that may want to be used, instead, it is a generic enough framework that any differentiable loss function can be used.

    Weak Learner
    Decision trees are used as the weak learner in gradient boosting.

    Specifically regression trees are used that output real values for splits and whose output can be added together, allowing subsequent models outputs to be added and “correct” the residuals in the predictions.

    Trees are constructed in a greedy manner, choosing the best split points based on purity scores like Gini or to minimize the loss. Initially, such as in the case of AdaBoost, very short decision trees were used that only had a single split, called a decision stump. Larger trees can be used generally with 4-to-8 levels.

    It is common to constrain the weak learners in specific ways, such as a maximum number of layers, nodes, splits or leaf nodes. This is to ensure that the learners remain weak, but can still be constructed in a greedy manner.

    Additive Model
    Trees are added one at a time, and existing trees in the model are not changed.

    A gradient descent procedure is used to minimize the loss when adding trees. Traditionally, gradient descent is used to minimize a set of parameters, such as the coefficients in a regression equation or weights in a neural network. After calculating error or loss, the weights are updated to minimize that error.

    Instead of parameters, we have weak learner sub-models or more specifically decision trees. After calculating the loss, to perform the gradient descent procedure, we must add a tree to the model that reduces the loss (i.e. follow the gradient). We do this by parameterizing the tree, then modify the parameters of the tree and move in the right direction by reducing the residual loss.

    Generally this approach is called functional gradient descent or gradient descent with functions. The output for the new tree is then added to the output of the existing sequence of trees in an effort to correct or improve the final output of the model.

    A fixed number of trees are added or training stops once loss reaches an acceptable level or no longer improves on an external validation dataset.


24. 
    Difference between AdaBoost and XGBoost.
    Both methods combine weak learners into one strong learner. For example, one decision tree is a weak learner, and an emsemble of them would be a random forest model, which is a strong learner.

    Both methods in the learning process will increase the ensemble of weak-trainers, adding new weak learners to the ensemble at each training iteration, i.e. in the case of the forest, the forest will grow with new trees. The only difference between AdaBoost and XGBoost is how the ensemble is replenished.

    AdaBoost works by weighting the observations, putting more weight on difficult to classify instances and less on those already handled well. New weak learners are added sequentially that focus their training on the more difficult patterns. AdaBoost at each iteration changes the sample weights in the sample. It raises the weight of the samples in which more mistakes were made. The sample weights vary in proportion to the ensemble error. We thereby change the probabilistic distribution of samples - those that have more weight will be selected more often in the future. It is as if we had accumulated samples on which more mistakes were made and would use them instead of the original sample. In addition, in AdaBoost, each weak learner has its own weight in the ensemble (alpha weight) - this weight is higher, the “smarter” this weak learner is, i.e. than the learner least likely to make mistakes.

    XGBoost does not change the selection or the distribution of observations at all. XGBoost builds the first tree (weak learner), which will fit the observations with some prediction error. A second tree (weak learner) is then added to correct the errors made by the existing model. Errors are minimized using a gradient descent algorithm. Regularization can also be used to penalize more complex models through both Lasso and Ridge regularization.

    In short, AdaBoost- reweighting examples. Gradient boosting - predicting the loss function of trees. Xgboost - the regularization term was added to the loss function (depth + values ​​in leaves).

31. Knowledge Distillation
    It is the process by which a considerably larger model is able to transfer its knowledge to a smaller one. Applications include NLP and object detection allowing for less powerful hardware to make good inferences without significant loss of accuracy.

    Example: model compression which is used to compress the knowledge of multiple models into a single neural network.


 32.   What is Pseudo Labeling?
    Pseudo labeling is the process of adding confident predicted test data to your training data. Pseudo labeling is a 5 step process. (1) Build a model using training data. (2) Predict labels for an unseen test dataset. (3) Add confident predicted test observations to our training data (4) Build a new model using combined data. And (5) use your new model to predict the test data and submit to Kaggle. Here is a pictorial explanation using sythetic 2D data.

    Step 1 - Build first model
    Given 50 training observations (25 target=1 yellow points, 25 target=0 blue points) build a model using QDA. Notice how QDA calculates the two multivariate Gaussian distributions that the target=1 and target=0 were drawn from. QDA's approximation is represented as ellipses of 1, 2, 3 standard deviations for each distribution.image

    Step 2 - Predict test data
    Using our model (ellipses), predict the target of 50 unknown data points. The bottom picture shows the decisions made by our classifier.image

    image

    Step 3 and 4 - Add pseudo label data and build second model
    Add all predictions with Pr(y=1|x)>0.99 and Pr(y=0|x)>0.99 to our training data. Then train a new model using the combined 90 points with QDA. The red ellipses show QDA's new approximation of the two Gaussian distributions. This time QDA has found better ellipses then before.

    image

    Step 5 - Predict test data
    Finally use our more accurate QDA ellipses to predict test (a second time) and submit to Kaggle.

    Why does Pseudo Labeling work?
    When I first learned about pseudo labeling (from team Wizardry's 1st place solution here), I was surprised that it could increase a model's accuracy. How does training with unknown data that has been labeled by a model improve that same model? Doesn't the model already know the information? Because the model made those predictions.

    How pseudo labeling works is best understood with QDA. QDA works by using points in p-dimensional space to find hyper-ellipsoids, see here. With more points, QDA can better estimate the center and shape of each ellipsoid (and consequently make better preditions afterward).

    Pseudo labeling helps all types of models because all models can be visualized as finding shapes of target=1 and target=0 in p-dimensional space. See here for examples. More points allow for better estimation of shapes.




Q4. Why might it be preferable to include fewer predictors over many?
    Anmol Rajpurohit answers:

    Here are a few reasons why it might be a better idea to have fewer predictor variables rather than having many of them:

    Redundancy/Irrelevance:

    If you are dealing with many predictor variables, then the chances are high that there are hidden relationships between some of them, leading to redundancy. Unless you identify and handle this redundancy (by selecting only the non-redundant predictor variables) in the early phase of data analysis, it can be a huge drag on your succeeding steps.

    It is also likely that not all predictor variables are having a considerable impact on the dependent variable(s). You should make sure that the set of predictor variables you select to work on does not have any irrelevant ones – even if you know that data model will take care of them by giving them lower significance.

    Note: Redundancy and Irrelevance are two different notions –a relevant feature can be redundant due to the presence of other relevant feature(s).

    Overfitting:

    Even when you have a large number of predictor variables with no relationships between any of them, it would still be preferred to work with fewer predictors. The data models with large number of predictors (also referred to as complex models) often suffer from the problem of overfitting, in which case the data model performs great on training data, but performs poorly on test data.

    Productivity:

    Let’s say you have a project where there are a large number of predictors and all of them are relevant (i.e. have measurable impact on the dependent variable). So, you would obviously want to work with all of them in order to have a data model with very high success rate. While this approach may sound very enticing, practical considerations (such of amount of data available, storage and compute resources, time taken for completion, etc.) make it nearly impossible.

    Thus, even when you have a large number of relevant predictor variables, it is a good idea to work with fewer predictors (shortlisted through feature selection or developed through feature extraction). This is essentially similar to the Pareto principle, which states that for many events, roughly 80% of the effects come from 20% of the causes.

    Focusing on those 20% most significant predictor variables will be of great help in building data models with considerable success rate in a reasonable time, without needing non-practical amount of data or other resources.


    Training error & test error vs model complexity (Source: Posted on Quora by Sergul Aydore)

    Understandability:

    Models with fewer predictors are way easier to understand and explain. As the data science steps will be performed by humans and the results will be presented (and hopefully, used) by humans, it is important to consider the comprehensive ability of human brain. This is basically a trade-off – you are letting go of some potential benefits to your data model’s success rate, while simultaneously making your data model easier to understand and optimize.

    This factor is particularly important if at the end of your project you need to present your results to someone, who is interested in not just high success rate, but also in understanding what is happening “under the hood”.


METRICS:
    With the help of a confusion matrix, we can calculate important performance measures:
    True Positive Rate (TPR) or Hit Rate or Recall or Sensitivity = TP / (TP + FN)
    False Positive Rate(FPR) or False Alarm Rate = 1 - Specificity = 1 - (TN / (TN + FP))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Error Rate = 1 – accuracy or (FP + FN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    F-measure: 2 / ( (1 / Precision) + (1 / Recall) )
    ROC (Receiver Operating Characteristics) = plot of FPR vs TPR
    AUC (Area Under the Curve)
    Kappa statistics

    All of these measures should be used with domain skills and balanced, as, for example, if you only 
    get a higher TPR in predicting patients who don’t have cancer, it will not help at all in diagnosing cancer.

    In the same example of cancer diagnosis data, if only 2% or less of the patients have cancer, 
    then this would be a case of class imbalance, as the percentage of cancer patients is 
    very small compared to rest of the population. There are main 2 approaches to handle this issue:

    Use of a cost function: In this approach, a cost associated with misclassifying data is 
    evaluated with the help of a cost matrix (similar to the confusion matrix, but more 
    concerned with False Positives and False Negatives). The main aim is to reduce 
    the cost of misclassifying. The cost of a False Negative is always more than 
    the cost of a False Positive. e.g. wrongly predicting a cancer patient to be cancer-free 
    is more dangerous than wrongly predicting a cancer-free patient to have cancer.
    Total Cost = Cost of FN * Count of FN + Cost of FP * Count of FP

    Use of different sampling methods: In this approach, you can use over-sampling, 
    under-sampling, or hybrid sampling. In over-sampling, minority class observations are 
    replicated to balance the data. Replication of observations leading to overfitting, 
    causing good accuracy in training but less accuracy in unseen data. In under-sampling, 
    the majority class observations are removed causing loss of information. It is helpful in 
    reducing processing time and storage, but only useful if you have a large data set.
    Find more about class imbalance here.

    If there are multiple classes in the target variable, then a confusion matrix of 
    dimensions equal to the number of classes is formed, and all performance measures 
    can be calculated for each of the classes. This is called a multiclass confusion 
    matrix. e.g. there are 3 classes X, Y, Z in the response variable, so 
    recall for each class will be calculated as below:

    Recall_X = TP_X/(TP_X+FN_X)

    Recall_Y = TP_Y/(TP_Y+FN_Y)

    Recall_Z = TP_Z/(TP_Z+FN_Z)


Q6. What are some ways I can make my model more robust to outliers?
    Thuy Pham answers:

    There are several ways to make a model more robust to outliers, from different points of view (data preparation or model building). An outlier in the question and answer is assumed being unwanted, unexpected, or a must-be-wrong value to the human’s knowledge so far (e.g. no one is 200 years old) rather than a rare event which is possible but rare.

    Outliers are usually defined in relation to the distribution. Thus outliers could be removed in the pre-processing step (before any learning step), by using standard deviations (for normality) or interquartile ranges (for not normal/unknown) as threshold levels.


    Outliers. Image source

    Moreover, data transformation (e.g. log transformation) may help if data have a noticeable tail. When outliers related to the sensitivity of the collecting instrument which may not precisely record small values, Winsorization may be useful. This type of transformation (named after Charles P. Winsor (1895–1951)) has the same effect as clipping signals (i.e. replaces extreme data values with less extreme values).  Another option to reduce the influence of outliers is using mean absolute difference rather mean squared error.

    For model building, some models are resistant to outliers (e.g. tree-based approaches) or non-parametric tests. Similar to the median effect, tree models divide each node into two in each split. Thus, at each split, all data points in a bucket could be equally treated regardless of extreme values they may have. The study [Pham 2016] proposed a detection model that incorporates interquartile information of data to predict outliers of the data.


Several methods can be used to avoid "overfitting" the data:

    Try to find the simplest possible hypothesis
    Regularization (adding a penalty for complexity)
    Randomization Testing (randomize the class variable, try your method on this data - if it find the same strong results, something is wrong)
    Nested cross-validation  (do feature selection on one level, then run entire method in cross-validation on outer level)
    Adjusting the False Discovery Rate
    Using the reusable holdout method - a breakthrough approach proposed in 2015



Q12. In unsupervised learning, if a ground truth about a dataset is unknown, how can we determine the most useful number of clusters to be?
    
    The Elbow Method

    The elbow method is often the best place to state, and is especially useful due to its ease 
    of explanation and verification via visualization. The elbow method is interested in 
    explaining variance as a function of cluster numbers (the k in k-means). By plotting 
    the percentage of variance explained against k, the first N clusters should add significant 
    information, explaining variance; yet, some eventual value of k will result in a much less 
    significant gain in information, and it is at this point that the graph will provide a 
    noticeable angle. This angle will be the optimal number of clusters, from the perspective of the elbow method,

    It should be self-evident that, in order to plot this variance against varying numbers of clusters, 
    varying numbers of clusters must be tested. Successive complete iterations of the clustering 
    method must be undertaken, after which the results can be plotted and compared.

    Elbow method
    Image source.
    The Silhouette Method

    The silhouette method measures the similarity of an object to its own cluster -- 
    called cohesion -- when compared to other clusters -- called separation. The silhouette 
    value is the means for this comparison, which is a value of the range [-1, 1]; a value 
    close to 1 indicates a close relationship with objects in its own cluster, while a value 
    close to -1 indicates the opposite. A clustered set of data in a model producing mostly 
    high silhouette values is likely an acceptable and appropriate model.s



Gradient boosting vs Random Forest Tree:

    Random Forest vs Decision Trees
    As noted above, decision trees are fraught with problems. A tree generated from 99 data points might differ significantly from a tree generated with just one different data point. If there was a way to generate a very large number of trees, averaging out their solutions, then you'll likely get an answer that is going to be very close to the true answer. Enter the random forest—a collection of decision trees with a single, aggregated result. Random forests are commonly reported as the most accurate learning algorithm. 

    Random forests reduce the variance seen in decision trees by:

    Using different samples for training,
    Specifying random feature subsets, 
    Building and combining small (shallow) trees.
    A single decision tree is a weak predictor, but is relatively fast to build. More trees give you a more robust model and prevent overfitting. However, the more trees you have, the slower the process. Each tree in the forest has to be generated, processed, and analyzed. In addition, the more features you have, the slower the process (which can sometimes take hours or even days); Reducing the set of features can dramatically speed up the process.

    Another distinct difference between a decision tree and random forest is that while a decision tree is easy to read—you just follow the path and find a result—a random forest is a tad more complicated to interpret. There are a slew of articles out there designed to help you read the results from random forests (like this one), but in comparison to decision trees, the learning curve is steep.

    Random Forest vs Gradient Boosting
    Like random forests, gradient boosting is a set of decision trees. The two main differences are:

    How trees are built: random forests builds each tree independently while gradient boosting builds one tree at a time. This additive model (ensemble) works in a forward stage-wise manner, introducing a weak learner to improve the shortcomings of existing weak learners. 
    Combining results: random forests combine results at the end of the process (by averaging or "majority rules") while gradient boosting combines results along the way.
    If you carefully tune parameters, gradient boosting can result in better performance than random forests. However, gradient boosting may not be a good choice if you have a lot of noise, as it can result in overfitting. They also tend to be harder to tune than random forests.

    Random forests and gradient boosting each excel in different areas. Random forests perform well for multi-class object detection and bioinformatics, which tends to have a lot of statistical noise. Gradient Boosting performs well when you have unbalanced data such as in real time risk assessment.



Q8. What is the difference between Point Estimates and Confidence Interval?
    Point Estimation gives us a particular value as an estimate of a population parameter. Method of Moments and Maximum Likelihood estimator methods are used to derive Point Estimators for population parameters.

    A confidence interval gives us a range of values which is likely to contain the population parameter. The confidence interval is generally preferred, as it tells us how likely this interval is to contain the population parameter. This likeliness or probability is called Confidence Level or Confidence coefficient and represented by 1 — alpha, where alpha is the level of significance.


Q11. In any 15-minute interval, there is a 20% probability that you will see at least one shooting star. What is the proba­bility that you see at least one shooting star in the period of an hour?

        Probability of not seeing any shooting star in 15 minutes is

        =   1 – P( Seeing one shooting star )
        =   1 – 0.2          =    0.8

        Probability of not seeing any shooting star in the period of one hour

        =   (0.8) ^ 4        =    0.4096

        Probability of seeing at least one shooting star in the one hour

        =   1 – P( Not seeing any star )
        =   1 – 0.4096     =    0.5904


Q14. A jar has 1000 coins, of which 999 are fair and 1 is double headed. Pick a coin at random, and toss it 10 times. Given that you see 10 heads, what is the probability that the next toss of that coin is also a head?

    There are two ways of choosing the coin. One is to pick a 
    fair coin and the other is to pick the one with two heads.

    Probability of selecting fair coin = 999/1000 = 0.999
    Probability of selecting unfair coin = 1/1000 = 0.001

    Selecting 10 heads in a row = Selecting fair coin * Getting 10 heads  +  Selecting an unfair coin

    P (A)  =  0.999 * (1/2)^5  =  0.999 * (1/1024)  =  0.000976
    P (B)  =  0.001 * 1  =  0.001
    P( A / A + B )  = 0.000976 /  (0.000976 + 0.001)  =  0.4939
    P( B / A + B )  = 0.001 / 0.001976  =  0.5061

    Probability of selecting another head = P(A/A+B) * 0.5 + P(B/A+B) * 1 = 0.4939 * 0.5 + 0.5061  =  0.7531



Q21.  What Are Confounding Variables?

    In statistics, a confounder is a variable that influences both the dependent variable and independent variable.

    For example, if you are researching whether a lack of exercise leads to weight gain,

    lack of exercise = independent variable

    weight gain = dependent variable.

    A confounding variable here would be any other variable that affects both of these variables, such as the age of the subject.


Q23. What is Survivorship Bias?

    It is the logical error of focusing aspects that support surviving some process and casually overlooking those that did not work because of their lack of prominence. This can lead to wrong conclusions in numerous different means.

ROC CURVE PLOTTING:


    A receiver operating characteristic curve, or ROC curve, is a graphical plot 
    that illustrates the diagnostic ability of a binary classifier system as its 
    discrimination threshold is varied. The method was developed for operators of 
    military radar receivers, which is why it is so named.

    The ROC curve is created by plotting the true positive rate (TPR) against the false 
    positive rate (FPR) at various threshold settings. The true-positive rate is also 
    known as sensitivity, recall or probability of detection[8] in machine learning. 
    The false-positive rate is also known as probability of false alarm[8] and can be 
    calculated as (1 − specificity). It can also be thought of as a plot of the power 
    as a function of the Type I Error of the decision rule (when the performance 
    is calculated from just a sample of the population, it can be thought of as 
    estimators of these quantities). The ROC curve is thus the sensitivity or recall as a 
    function of fall-out. In general, if the probability distributions 
    for both detection and false alarm are known, the ROC curve can be generated by 
    plotting the cumulative distribution function (area under the probability 
    distribution from {\displaystyle -\infty }-\infty  to the discrimination threshold) 
    of the detection probability in the y-axis versus the cumulative distribution 
    function of the false-alarm probability on the x-axis.

Q26. What is TF/IDF vectorization?

    TF–IDF is short for term frequency-inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining.

    The TF–IDF value increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.

KMEANS -> how to choose K ->
    Elbow method, by taking square distance sum of each cluster center to surrounding points in cluster 
    and measure homeogenity
    Or use Hierarchal clustering to determine K.

Q65. What cross-validation technique would you use on a time series data set?

    fold 1: training[1], test[2]

    fold 1: training[1 2], test[3]

    fold 1: training[1 2 3], test[4]

    fold 1: training[1 2 3 4], test[5]


Q68. If you are having 4GB RAM in your machine and you want to train your model on 10GB data set. How would you go about this problem? Have you ever faced this kind of problem in your machine learning/data science experience so far?
    First of all, you have to ask which ML model you want to train.

    For Neural networks: Batch size with Numpy array will work.

    Steps:

    Load the whole data in the Numpy array. Numpy array has a property to create a mapping of the complete data set, it doesn’t load complete data set in memory.

    You can pass an index to Numpy array to get required data.

    Use this data to pass to the Neural network.

    Have a small batch size.

    For SVM: Partial fit will work

    Steps:

    Divide one big data set in small size data sets.

    Use a partial fit method of SVM, it requires a subset of the complete data set.

    Repeat step 2 for other subsets.


What is a box cox transformation?
    https://www.statisticshowto.com/box-cox-transformation/#:~:text=What%20is%20a%20Box%20Cox,a%20broader%20number%20of%20tests.


    Common Box-Cox Transformations
    Lambda value (λ)	Transformed data (Y’)
    -3	Y-3 = 1/Y3
    -2	Y-2 = 1/Y2
    -1	Y-1 = 1/Y1
    -0.5	Y-0.5 = 1/(√(Y))
    0	log(Y)**
    0.5	Y0.5 = √(Y)
    1	Y1 = Y
    2	Y2
    3	Y3


2. Explain what a long-tailed distribution is and provide three examples of relevant phenomena that have long tails. Why are they important in classification and regression problems?
    Example of a long tail distribution
    A long-tailed distribution is a type of heavy-tailed distribution that has a tail (or tails) that drop off gradually and asymptotically.
    3 practical examples include the power law, the Pareto principle (more commonly known as the 80–20 rule), and product sales (i.e. best selling products vs others).
    It’s important to be mindful of long-tailed distributions in classification and regression problems because the least frequently occurring values make up the majority of the population. This can ultimately change the way that you deal with outliers, and it also conflicts with some machine learning techniques with the assumption that the data is normally distributed.


3. What is the Central Limit Theorem? Explain it. Why is it important?
    Statistics How To provides the best definition of CLT, which is:
    “The central limit theorem states that the sampling distribution of the sample 
    mean approaches a normal distribution as the sample size 
    gets larger no matter what the shape of the population distribution.” [1]
    
    The central limit theorem is important because it is used in hypothesis 
    testing and also to calculate confidence intervals.


4. What is the statistical power?
    ‘Statistical power’ refers to the power of a binary hypothesis, which is the probability that 
    the test rejects the null hypothesis given that the alternative hypothesis is true. [2]

    Power = P(reject Null | alternative is True)


5. Explain selection bias (with regard to a dataset, not variable selection). Why is it important? How can data management procedures such as missing data handling make it worse?
    Selection bias is the phenomenon of selecting individuals, groups or data for analysis in such a way that proper randomization is not achieved, ultimately resulting in a sample that is not representative of the population.
    Understanding and identifying selection bias is important because it can significantly skew results and provide false insights about a particular population group.
    Types of selection bias include:


    Types of selection bias include:
    sampling bias: a biased sample caused by non-random sampling
    time interval: selecting a specific time frame that supports the desired conclusion. e.g. conducting a sales analysis near Christmas.
    exposure: includes clinical susceptibility bias, protopathic bias, indication bias. Read more here.
    data: includes cherry-picking, suppressing evidence, and the fallacy of incomplete evidence.
    attrition: attrition bias is similar to survivorship bias, where only those that ‘survived’ a long process are included in an analysis, or failure bias, where those that ‘failed’ are only included
    observer selection: related to the Anthropic principle, which is a philosophical consideration that any data we collect about the universe is filtered by the fact that, in order for it to be observable, it must be compatible with the conscious and sapient life that observes it. [3]

    Handling missing data can make selection bias worse because different methods impact the data in different ways. 
    For example, if you replace null values with the mean of the data, you adding bias in the sense that you’re assuming that 
    the data is not as spread out as it might actually be.


7. Is mean imputation of missing data acceptable practice? Why or why not?
    Mean imputation is the practice of replacing null values 
    in a data set with the mean of the data.
    
    Mean imputation is generally bad practice because it doesn’t take into account 
    feature correlation. For example, imagine we have a table showing age and 
    fitness score and imagine that an eighty-year-old has a missing fitness score. 
    If we took the average fitness score from an age range of 15 to 80, then 
    the eighty-year-old will appear to have a much higher fitness score that he actually should.

    Second, mean imputation reduces the variance of the data and increases bias in our data. 
    This leads to a less accurate model and a narrower 
    confidence interval due to a smaller variance.



8. What is an outlier? Explain how you might screen for outliers and what would you do 
   if you found them in your dataset. Also, explain what an inlier is and how you might screen 
   for them and what would you do if you found them in your dataset.
    
    An outlier is a data point that differs significantly from other observations.
    Depending on the cause of the outlier, they can be bad from a machine learning perspective 
    because they can worsen the accuracy of a model. If the outlier is caused by a measurement error, 
    it’s important to remove them from the dataset. There are a couple of ways to identify outliers:
    Z-score/standard deviations: if we know that 99.7% of data in a data set lie within three standard deviations, 
    then we can calculate the size of one standard deviation, multiply it by 3, and 
    identify the data points that are outside of this range. Likewise, we can calculate the z-score of a 
    given point, and if it’s equal to +/- 3, then it’s an outlier.

    Note: that there are a few contingencies that need to be considered when using 
    this method; the data must be normally distributed, this is not applicable for 
    small data sets, and the presence of too many outliers can throw off z-score.


    Interquartile Range (IQR): IQR, the concept used to build boxplots, can also be 
    used to identify outliers. The IQR is equal to the difference between the 
    3rd quartile and the 1st quartile. You can then identify if a point is an 
    outlier if it is less than Q1–1.5*IRQ or greater than Q3 + 1.5*IQR. 
    This comes to approximately 2.698 standard deviations.


    Other methods include DBScan clustering, Isolation Forests, and Robust Random Cut Forests.

    An inlier is a data observation that lies within the rest of the dataset and is 
    unusual or an error. Since it lies in the dataset, it is typically harder to 
    identify than an outlier and requires external data to identify them. Should 
    you identify any inliers, you can simply remove them from the dataset to address them.



9. How do you handle missing data? What imputation techniques do you recommend?
    There are several ways to handle missing data:
    Delete rows with missing data
    Mean/Median/Mode imputation
    Assigning a unique value
    Predicting the missing values
    Using an algorithm which supports missing values, like random forests

    The best method is to delete rows with missing data as it ensures that no 
    bias or variance is added or removed, and ultimately results in a robust and 
    ccurate model. However, this is only recommended if there’s a lot of data to 
    start with and the percentage of missing values is low.

10. You have data on the duration of calls to a call center. Generate a plan for how you 
    would code and analyze these data. Explain a plausible scenario for what the distribution of these 
    durations might look like. How could you test, even graphically, whether your expectations are borne out?

    First I would conduct EDA — Exploratory Data Analysis to clean, explore, and understand my data. See my article on EDA here. As part of my EDA, I could compose a histogram of the duration of calls to see the underlying distribution.
    My guess is that the duration of calls would follow a lognormal distribution (see below). The reason that I believe it’s positively skewed is because the lower end is limited to 0 since a call can’t be negative seconds. However, on the upper end, it’s likely for there to be a small proportion of calls that are extremely long relatively.
    Image for post
    Lognormal Distribution Example
    You could use a QQ plot to confirm whether the duration of calls follows a 
    lognormal distribution or not. See here to learn more about QQ plots.



12. You are compiling a report for user content uploaded every month and notice a spike in uploads in October. In particular, a spike in picture uploads. What might you think is the cause of this, and how would you test it?

    The method of testing depends on the cause of the spike, but you would conduct hypothesis
    testing to determine if the inferred cause is the actual cause.

13. You’re about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it’s raining. Each of your friends has a 2/3 chance of telling you the truth and a 1/3 chance of messing with you by lying. All 3 friends tell you that “Yes” it is raining. What is the probability that it’s actually raining in Seattle?
    You can tell that this question is related to Bayesian theory because of the last statement which essentially follows the structure, “What is the probability A is true given B is true?” Therefore we need to know the probability of it raining in London on a given day. Let’s assume it’s 25%.
    P(A) = probability of it raining = 25%
    P(B) = probability of all 3 friends say that it’s raining
    P(A|B) probability that it’s raining given they’re telling that it is raining
    P(B|A) probability that all 3 friends say that it’s raining given it’s raining = (2/3)³ = 8/27
    Step 1: Solve for P(B)
    P(A|B) = P(B|A) * P(A) / P(B), can be rewritten as
    P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)
    P(B) = (2/3)³ * 0.25 + (1/3)³ * 0.75 = 0.25*8/27 + 0.75*1/27
    Step 2: Solve for P(A|B)
    P(A|B) = 0.25 * (8/27) / ( 0.25*8/27 + 0.75*1/27)
    P(A|B) = 8 / (8 + 3) = 8/11
    Therefore, if all three friends say that it’s raining, then there’s an 8/11 chance that it’s actually raining.

14. There’s one box — has 12 black and 12 red cards, 2nd box has 24 black and 24 red; if you want to draw 2 cards at random from one of the 2 boxes, which box has the higher probability of getting the same color? Can you tell intuitively why the 2nd box has a higher probability
    The box with 24 red cards and 24 black cards has a higher probability of getting two cards of the same color. Let’s walk through each step.
    Let’s say the first card you draw from each deck is a red Ace.
    This means that in the deck with 12 reds and 12 blacks, there’s now 11 reds and 12 blacks. Therefore your odds of drawing another red are equal to 11/(11+12) or 11/23.
    In the deck with 24 reds and 24 blacks, there would then be 23 reds and 24 blacks. Therefore your odds of drawing another red are equal to 23/(23+24) or 23/47.
    Since 23/47 > 11/23, the second deck with more cards has a higher probability of getting the same two cards.


17. Give examples of data that does not have a Gaussian distribution, nor log-normal.
    Any type of categorical data won’t have a gaussian distribution or lognormal distribution.
    Exponential distributions — eg. the amount of time that a car battery lasts or the amount of time until an earthquake occurs.


18. What is root cause analysis? How to identify a cause vs. a correlation? Give examples
    Root cause analysis: a method of problem-solving used for identifying the root cause(s) of a problem [5]

    You can test for causation using hypothesis testing or A/B testing.

19. Give an example where the median is a better measure than the mean
    When there are a number of outliers that positively or negatively skew the data.


22. How do you calculate the needed sample size?

    MARGIN OF ERROR = t * S/sqrt(n)
    MARGIN OF ERROR = z * sigma/sqrt(n)

    You can use the margin of error (ME) formula to determine the desired sample size.
    t/z = t/z score used to calculate the confidence interval
    ME = the desired margin of error
    S = sample standard deviation


23. When you sample, what bias are you inflicting?
    Potential biases include the following:
    Sampling bias: a biased sample caused by non-random sampling
    Under coverage bias: sampling too few observations
    Survivorship bias: error of overlooking observations that did not make it past a form of selection process.


24. How do you control for biases?
    There are many things that you can do to control and minimize bias. Two common things include randomization, where participants are assigned by chance, and random sampling, sampling in which each member has an equal probability of being chosen.


25. What are confounding variables?
    A confounding variable, or a confounder, is a variable that influences both the dependent variable and the independent variable, causing a spurious association, a mathematical relationship in which two or more variables are associated but not causally related.

26. What is A/B testing?
    A/B testing is a form of hypothesis testing and two-sample hypothesis testing to compare two versions, the control and variant, of a single variable. It is commonly used to improve and optimize user experience and marketing.


27) 1. Why do you use feature selection?
    Feature selection is the process of selecting a subset of relevant features for use in model construction. Feature selection is itself useful, but it mostly acts as a filter, muting out features that aren’t useful in addition to your existing features. Feature selection methods aid you in your mission to create an accurate predictive model. They help you by choosing features that will give you as good or better accuracy whilst requiring less data. Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model. Fewer attributes is desirable because it reduces the complexity of the model, and a simpler model is simpler to understand and explain.

    Filter Methods
    Filter feature selection methods apply a statistical measure to assign a scoring to each feature. The features are ranked by the score and either selected to be kept or removed from the dataset. The methods are often univariate and consider the feature independently, or with regard to the dependent variable. Some examples of some filter methods include the Chi squared test, information gain and correlation coefficient scores.

    Embedded Methods
    Embedded methods learn which features best contribute to the accuracy of the model while the model is being created. The most common type of embedded feature selection methods are regularization methods. Regularization methods are also called penalization methods that introduce additional constraints into the optimization of a predictive algorithm (such as a regression algorithm) that bias the model toward lower complexity (fewer coefficients). Examples of regularization algorithms are the LASSO, Elastic Net and Ridge Regression.

    Misleading
    Including redundant attributes can be misleading to modeling algorithms. Instance-based methods such as k-nearest neighbor use small neighborhoods in the attribute space to determine classification and regression predictions. These predictions can be greatly skewed by redundant attributes.

    Overfitting
    Keeping irrelevant attributes in your dataset can result in overfitting. Decision tree algorithms like C4.5 seek to make optimal spits in attribute values. Those attributes that are more correlated with the prediction are split on first. Deeper in the tree less relevant and irrelevant attributes are used to make prediction decisions that may only be beneficial by chance in the training dataset. This overfitting of the training data can negatively affect the modeling power of the method and cripple the predictive accuracy.

28) Explain what regularization is and why it is useful.
    Regularization is the process of adding a tuning parameter to a model to induce smoothness in order to prevent overfitting.

    This is most often done by adding a constant multiple to an existing weight vector. This constant is often either the L1 (Lasso) or L2 (ridge), but can in actuality can be any norm. The model predictions should then minimize the mean of the loss function calculated on the regularized training set.

    It is well known, as explained by others, that L1 regularization helps perform feature selection in sparse feature spaces, and that is a good practical reason to use L1 in some situations. However, beyond that particular reason I have never seen L1 to perform better than L2 in practice. If you take a look at LIBLINEAR FAQ on this issue you will see how they have not seen a practical example where L1 beats L2 and encourage users of the library to contact them if they find one. Even in a situation where you might benefit from L1's sparsity in order to do feature selection, using L2 on the remaining variables is likely to give better results than L1 by itself.



4.  How would you validate a model you created to generate a predictive model of a quantitative outcome variable 
    using multiple regression?
    Proposed methods for model validation:

    If the values predicted by the model are far outside of the response variable range, this would immediately indicate poor estimation or model inaccuracy.
    If the values seem to be reasonable, examine the parameters; any of the following would indicate poor estimation or multi-collinearity: opposite signs of expectations, unusually large or small values, or observed inconsistency when the model is fed new data.
    Use the model for prediction by feeding it new data, and use the coefficient of determination (R squared) as a model validity measure.
    Use data splitting to form a separate dataset for estimating model parameters, and another for validating predictions.
    Use jackknife resampling if the dataset contains a small number of instances, and measure validity with R squared and mean squared error (MSE).


5. 
    Explain what precision and recall are. How do they relate to the ROC curve?
    Calculating precision and recall is actually quite easy. Imagine there are 100 positive cases among 10,000 cases. You want to predict which ones are positive, and you pick 200 to have a better chance of catching many of the 100 positive cases. You record the IDs of your predictions, and when you get the actual results you sum up how many times you were right or wrong. There are four ways of being right or wrong:

    TN / True Negative: case was negative and predicted negative
    TP / True Positive: case was positive and predicted positive
    FN / False Negative: case was positive but predicted negative
    FP / False Positive: case was negative but predicted positive
    alt text

    Now, your boss asks you three questions:

    What percent of your predictions were correct? You answer: the "accuracy" was (9,760+60) out of 10,000 = 98.2%
    What percent of the positive cases did you catch? You answer: the "recall" was 60 out of 100 = 60%
    What percent of positive predictions were correct? You answer: the "precision" was 60 out of 200 = 30% See also a very good explanation of Precision and recall in Wikipedia.
    alt text

    ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION) and is commonly used to measure the performance of binary classifiers. However, when dealing with highly skewed datasets, Precision-Recall (PR) curves give a more representative picture of performance. Remember, a ROC curve represents a relation between sensitivity (RECALL) and specificity(NOT PRECISION). Sensitivity is the other name for recall but specificity is not PRECISION.

    Recall/Sensitivity is the measure of the probability that your estimate is 1 given all the samples whose true class label is 1. It is a measure of how many of the positive samples have been identified as being positive. Specificity is the measure of the probability that your estimate is 0 given all the samples whose true class label is 0. It is a measure of how many of the negative samples have been identified as being negative.

    PRECISION on the other hand is different. It is a measure of the probability that a sample is a true positive class given that your classifier said it is positive. It is a measure of how many of the samples predicted by the classifier as positive is indeed positive. Note here that this changes when the base probability or prior probability of the positive class changes. Which means PRECISION depends on how rare is the positive class. In other words, it is used when positive class is more interesting than the negative class.

    Sensitivity also known as the True Positive rate or Recall is calculated as, Sensitivity = TP / (TP + FN). Since the formula doesn’t contain FP and TN, Sensitivity may give you a biased result, especially for imbalanced classes. In the example of Fraud detection, it gives you the percentage of Correctly Predicted Frauds from the pool of Actual Frauds pool of Actual Non-Frauds.
    Specificity, also known as True Negative Rate is calculated as, Specificity = TN / (TN + FP). Since the formula does not contain FN and TP, Specificity may give you a biased result, especially for imbalanced classes. In the example of Fraud detection, it gives you the percentage of Correctly Predicted Non-Frauds from the pool of Actual Frauds pool of Actual Non-Frauds
    Assessing and Comparing Classifier Performance with ROC Curves


7. 
    How do you deal with unbalanced binary classification?
    Imbalanced data typically refers to a problem with classification problems where the classes are not represented equally. For example, you may have a 2-class (binary) classification problem with 100 instances (rows). A total of 80 instances are labeled with Class-1 and the remaining 20 instances are labeled with Class-2.

    This is an imbalanced dataset and the ratio of Class-1 to Class-2 instances is 80:20 or more concisely 4:1. You can have a class imbalance problem on two-class classification problems as well as multi-class classification problems. Most techniques can be used on either. The remaining discussions will assume a two-class classification problem because it is easier to think about and describe.

    Can You Collect More Data?
    A larger dataset might expose a different and perhaps more balanced perspective on the classes. More examples of minor classes may be useful later when we look at resampling your dataset.
    Try Changing Your Performance Metric
    Accuracy is not the metric to use when working with an imbalanced dataset. We have seen that it is misleading. From that post, I recommend looking at the following performance measures that can give more insight into the accuracy of the model than traditional classification accuracy:
    Confusion Matrix: A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).
    Precision: A measure of a classifiers exactness. Precision is the number of True Positives divided by the number of True Positives and False Positives. Put another way, it is the number of positive predictions divided by the total number of positive class values predicted. It is also called the Positive Predictive Value (PPV). Precision can be thought of as a measure of a classifiers exactness. A low precision can also indicate a large number of False Positives.
    Recall: A measure of a classifiers completeness. Recall is the number of True Positives divided by the number of True Positives and the number of False Negatives. Put another way it is the number of positive predictions divided by the number of positive class values in the test data. It is also called Sensitivity or the True Positive Rate. Recall can be thought of as a measure of a classifiers completeness. A low recall indicates many False Negatives.
    F1 Score (or F-score): A weighted average of precision and recall. I would also advise you to take a look at the following:
    Kappa (or Cohen’s kappa): Classification accuracy normalized by the imbalance of the classes in the data. ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.
    Try Resampling Your Dataset
    You can add copies of instances from the under-represented class called over-sampling (or more formally sampling with replacement)
    You can delete instances from the over-represented class, called under-sampling.
    Try Different Algorithms
    Try Penalized Models
    You can use the same algorithms but give them a different perspective on the problem. Penalized classification imposes an additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model to pay more attention to the minority class. Often the handling of class penalties or weights are specialized to the learning algorithm. There are penalized versions of algorithms such as penalized-SVM and penalized-LDA. Using penalization is desirable if you are locked into a specific algorithm and are unable to resample or you’re getting poor results. It provides yet another way to “balance” the classes. Setting up the penalty matrix can be complex. You will very likely have to try a variety of penalty schemes and see what works best for your problem.
    Try a Different Perspective
    Taking a look and thinking about your problem from these perspectives can sometimes shame loose some ideas. Two you might like to consider are anomaly detection and change detection.


8. 
    What is statistical power?
    Statistical power or sensitivity of a binary hypothesis test is the probability that the test correctly rejects the null hypothesis (H0) when the alternative hypothesis (H1) is true.

    It can be equivalently thought of as the probability of accepting the alternative hypothesis (H1) when it is true—that is, the ability of a test to detect an effect, if the effect actually exists.

    To put in another way, Statistical power is the likelihood that a study will detect an effect when the effect is present. The higher the statistical power, the less likely you are to make a Type II error (concluding there is no effect when, in fact, there is).

    A type I error (or error of the first kind) is the incorrect rejection of a true null hypothesis. Usually a type I error leads one to conclude that a supposed effect or relationship exists when in fact it doesn't. Examples of type I errors include a test that shows a patient to have a disease when in fact the patient does not have the disease, a fire alarm going on indicating a fire when in fact there is no fire, or an experiment indicating that a medical treatment should cure a disease when in fact it does not.

    A type II error (or error of the second kind) is the failure to reject a false null hypothesis. Examples of type II errors would be a blood test failing to detect the disease it was designed to detect, in a patient who really has the disease; a fire breaking out and the fire alarm does not ring; or a clinical trial of a medical treatment failing to show that the treatment works when really it does.

    pregnant girl being told shes not pregnant -> type 2 error 
    man being told hes pregnant -> type 1 error


9. 
    What are bias and variance, and what are their relation to modeling data?
    Bias is how far removed a model's predictions are from correctness, while variance is the degree to which these predictions vary between model iterations.

    Bias is generally the distance between the model that you build on the training data (the best model that your model space can provide) and the “real model” (which generates data).

    Error due to Bias: Due to randomness in the underlying data sets, the resulting models will have a range of predictions. Bias measures how far off in general these models' predictions are from the correct value. The bias is error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

    Error due to Variance: The error due to variance is taken as the variability of a model prediction for a given data point. Again, imagine you can repeat the entire model building process multiple times. The variance is how much the predictions for a given point vary between different realizations of the model. The variance is error from sensitivity to small fluctuations in the training set.

    High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).

    Big dataset -> low variance
    Low dataset -> high variance
    Few features -> high bias, low variance
    Many features -> low bias, high variance
    Complicated model -> low bias
    Simplified model -> high bias
    Decreasing λ -> low bias
    Increasing λ -> low variance

    We can create a graphical visualization of bias and variance using a bulls-eye diagram. Imagine that the center of the target is a model that perfectly predicts the correct values. As we move away from the bulls-eye, our predictions get worse and worse. Imagine we can repeat our entire model building process to get a number of separate hits on the target. Each hit represents an individual realization of our model, given the chance variability in the training data we gather. Sometimes we will get a good distribution of training data so we predict very well and we are close to the bulls-eye, while sometimes our training data might be full of outliers or non-standard values resulting in poorer predictions. These different realizations result in a scatter of hits on the target. alt text

    As an example, using a simple flawed Presidential election survey as an example, errors in the survey are then explained through the twin lenses of bias and variance: selecting survey participants from a phonebook is a source of bias; a small sample size is a source of variance.

    Minimizing total model error relies on the balancing of bias and variance errors. Ideally, models are the result of a collection of unbiased data of low variance. Unfortunately, however, the more complex a model becomes, its tendency is toward less bias but greater variance; therefore an optimal model would need to consider a balance between these 2 properties.

    The statistical evaluation method of cross-validation is useful in both demonstrating the importance of this balance, as well as actually searching it out. The number of data folds to use -- the value of k in k-fold cross-validation -- is an important decision; the lower the value, the higher the bias in the error estimates and the less variance. alt text

    The most important takeaways are that bias and variance are two sides of an important trade-off when building models, and that even the most routine of statistical evaluation methods are directly reliant upon such a trade-off.

    We may estimate a model f̂ (X) of f(X) using linear regressions or another modeling technique. In this case, the expected squared prediction error at a point x is: Err(x)=E[(Y−f̂ (x))^2]

    This error may then be decomposed into bias and variance components: Err(x)=(E[f̂ (x)]−f(x))^2+E[(f̂ (x)−E[f̂ (x)])^2]+σ^2e Err(x)=Bias^2+Variance+Irreducible

    That third term, irreducible error, is the noise term in the true relationship that cannot fundamentally be reduced by any model. Given the true model and infinite data to calibrate it, we should be able to reduce both the bias and variance terms to 0. However, in a world with imperfect models and finite data, there is a tradeoff between minimizing the bias and minimizing the variance.

    That third term, irreducible error, is the noise term in the true relationship that cannot fundamentally be reduced by any model. Given the true model and infinite data to calibrate it, we should be able to reduce both the bias and variance terms to 0. However, in a world with imperfect models and finite data, there is a tradeoff between minimizing the bias and minimizing the variance.

    If a model is suffering from high bias, it means that model is less complex, to make the model more robust, we can add more features in feature space. Adding data points will reduce the variance.

    The bias–variance tradeoff is a central problem in supervised learning. Ideally, one wants to choose a model that both accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously. High-variance learning methods may be able to represent their training set well, but are at risk of overfitting to noisy or unrepresentative training data. In contrast, algorithms with high bias typically produce simpler models that don't tend to overfit, but may underfit their training data, failing to capture important regularities.

    Models with low bias are usually more complex (e.g. higher-order regression polynomials), enabling them to represent the training set more accurately. In the process, however, they may also represent a large noise component in the training set, making their predictions less accurate - despite their added complexity. In contrast, models with higher bias tend to be relatively simple (low-order or even linear regression polynomials), but may produce lower variance predictions when applied beyond the training set.

How to increase and decrease bias/variance:

    Approaches
    Dimensionality reduction and feature selection can decrease variance by simplifying models. Similarly, a larger training set tends to decrease variance. Adding features (predictors) tends to decrease bias, at the expense of introducing additional variance. Learning algorithms typically have some tunable parameters that control bias and variance, e.g.:

    (Generalized) linear models can be regularized to decrease their variance at the cost of increasing their bias.
    In artificial neural networks, the variance increases and the bias decreases with the number of hidden units. Like in GLMs, regularization is typically applied.
    In k-nearest neighbor models, a high value of k leads to high bias and low variance (see below).
    In Instance-based learning, regularization can be achieved varying the mixture of prototypes and exemplars.[
    In decision trees, the depth of the tree determines the variance. Decision trees are commonly pruned to control variance.
    One way of resolving the trade-off is to use mixture models and ensemble learning. For example, boosting combines many "weak" (high bias) models in an ensemble that has lower bias than the individual models, while bagging combines "strong" learners in a way that reduces their variance.

    Understanding the Bias-Variance Tradeoff


Formulas:

    True Positive Rate (TPR) or Recall or Sensitivity = TP / (TP + FN)
    Precision = TP / (TP + FP)
    False Positive Rate(FPR) or False Alarm Rate = 1 - Specificity = 1 - (TN / (TN + FP))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Error Rate = 1 – Accuracy
    F-measure = 2 / ((1 / Precision) + (1 / Recall)) = 2 * (precision * recall) / (precision + recall)
    ROC (Receiver Operating Characteristics) = plot of FPR vs TPR
    AUC (Area Under the [ROC] Curve)
    Performance measure across all classification thresholds. Treated as the probability that a model ranks a randomly chosen positive sample higher than negative

11. 
    What are some ways I can make my model more robust to outliers?
    There are several ways to make a model more robust to outliers, from different points of view (data preparation or model building). An outlier in the question and answer is assumed being unwanted, unexpected, or a must-be-wrong value to the human’s knowledge so far (e.g. no one is 200 years old) rather than a rare event which is possible but rare.

    Outliers are usually defined in relation to the distribution. Thus outliers could be removed in the pre-processing step (before any learning step), by using standard deviations (Mean +/- 2*SD), it can be used for normality. Or interquartile ranges Q1 - Q3, Q1 - is the "middle" value in the first half of the rank-ordered data set, Q3 - is the "middle" value in the second half of the rank-ordered data set. It can be used for not normal/unknown as threshold levels.

    Moreover, data transformation (e.g. log transformation) may help if data have a noticeable tail. When outliers related to the sensitivity of the collecting instrument which may not precisely record small values, Winsorization may be useful. This type of transformation (named after Charles P. Winsor (1895–1951)) has the same effect as clipping signals (i.e. replaces extreme data values with less extreme values). Another option to reduce the influence of outliers is using mean absolute difference rather mean squared error.

    For model building, some models are resistant to outliers (e.g. tree-based approaches) or non-parametric tests. Similar to the median effect, tree models divide each node into two in each split. Thus, at each split, all data points in a bucket could be equally treated regardless of extreme values they may have.


Parametrics ML algorithms:

    Parametric Machine Learning Algorithms
    Assumptions can greatly simplify the learning process, 
    but can also limit what can be learned. Algorithms that simplify 
    the function to a known form are called parametric machine learning algorithms.

    A learning model that summarizes data with a set of parameters of fixed 
    size (independent of the number of training examples) is called a 
    parametric model. No matter how much data you throw at a parametric model, 
    it won’t change its mind about how many parameters it needs.

    — Artificial Intelligence: A Modern Approach, page 737

    The algorithms involve two steps:

    Select a form for the function.
    Learn the coefficients for the function from the training data.
    An easy to understand functional form for the mapping function is a 
    line, as is used in linear regression:

    b0 + b1*x1 + b2*x2 = 0

    Where b0, b1 and b2 are the coefficients of the line that control the intercept and slope, and x1 and x2 are two input variables.

    Assuming the functional form of a line greatly simplifies the learning process. Now, all we need to do is estimate the coefficients of the line equation and we have a predictive model for the problem.

    Often the assumed functional form is a linear combination of the input variables and as such parametric machine learning algorithms are often also called “linear machine learning algorithms“.

    The problem is, the actual unknown underlying function may not be a linear function like a line. It could be almost a line and require some minor transformation of the input data to work right. Or it could be nothing like a line in which case the assumption is wrong and the approach will produce poor results.

    Some more examples of parametric machine learning algorithms include:

    Logistic Regression
    Linear Discriminant Analysis
    Perceptron
    Naive Bayes
    Simple Neural Networks
    Benefits of Parametric Machine Learning Algorithms:

    Simpler: These methods are easier to understand and interpret results.
    Speed: Parametric models are very fast to learn from data.
    Less Data: They do not require as much training data and can work well even if the fit to the data is not perfect.

    Limitations of Parametric Machine Learning Algorithms:
    Constrained: By choosing a functional form these methods are highly constrained to the specified form.
    Limited Complexity: The methods are more suited to simpler problems.
    Poor Fit: In practice the methods are unlikely to match the underlying mapping function.
    
Nonparametric Machine Learning Algorithms
    Algorithms that do not make strong assumptions about the form of the mapping function are called nonparametric machine learning algorithms. By not making assumptions, they are free to learn any functional form from the training data.

    Nonparametric methods are good when you have a lot of data and no prior knowledge, and when you don’t want to worry too much about choosing just the right features.

    — Artificial Intelligence: A Modern Approach, page 757

    Nonparametric methods seek to best fit the training data in constructing the mapping function, whilst maintaining some ability to generalize to unseen data. As such, they are able to fit a large number of functional forms.

    An easy to understand nonparametric model is the k-nearest neighbors algorithm that makes predictions based on the k most similar training patterns for a new data instance. The method does not assume anything about the form of the mapping function other than patterns that are close are likely to have a similar output variable.

    Some more examples of popular nonparametric machine learning algorithms are:

    k-Nearest Neighbors
    Decision Trees like CART and C4.5
    Support Vector Machines
    Benefits of Nonparametric Machine Learning Algorithms:

    Flexibility: Capable of fitting a large number of functional forms.
    Power: No assumptions (or weak assumptions) about the underlying function.
    Performance: Can result in higher performance models for prediction.
    Limitations of Nonparametric Machine Learning Algorithms:

    More data: Require a lot more training data to estimate the mapping function.
    Slower: A lot slower to train as they often have far more parameters to train.
    Overfitting: More of a risk to overfit the training data and it is harder to explain why specific predictions are made.



13. 
    Define variance
    Variance is the expectation of the squared deviation of a random variable from its mean. Informally, it measures how far a set of (random) numbers are spread out from their average value. The variance is the square of the standard deviation, the second central moment of a distribution, and the covariance of the random variable with itself.

    Var(X) = E[(X - m)^2], m=E[X]

    Variance is, thus, a measure of the scatter of the values of a random variable relative to its mathematical expectation.

16. 
    How would you find an anomaly in a distribution?
    Before getting started, it is important to establish some boundaries on the definition of an anomaly. Anomalies can be broadly categorized as:

    Point anomalies: A single instance of data is anomalous if it's too far off from the rest. Business use case: Detecting credit card fraud based on "amount spent."
    Contextual anomalies: The abnormality is context specific. This type of anomaly is common in time-series data. Business use case: Spending $100 on food every day during the holiday season is normal, but may be odd otherwise.
    Collective anomalies: A set of data instances collectively helps in detecting anomalies. Business use case: Someone is trying to copy data form a remote machine to a local host unexpectedly, an anomaly that would be flagged as a potential cyber attack.
    Best steps to prevent anomalies is to implement policies or checks that can catch them during the data collection stage. Unfortunately, you do not often get to collect your own data, and often the data you're mining was collected for another purpose. About 68% of all the data points are within one standard deviation from the mean. About 95% of the data points are within two standard deviations from the mean. Finally, over 99% of the data is within three standard deviations from the mean. When the value deviate too much from the mean, let’s say by ± 4σ, then we can considerate this almost impossible value as anomaly. (This limit can also be calculated using the percentile).

    Statistical methods
    Statistically based anomaly detection uses this knowledge to discover outliers. A dataset can be standardized by taking the z-score of each point. A z-score is a measure of how many standard deviations a data point is away from the mean of the data. Any data-point that has a z-score higher than 3 is an outlier, and likely to be an anomaly. As the z-score increases above 3, points become more obviously anomalous. A z-score is calculated using the following equation. A box-plot is perfect for this application.

    Metric method
    Judging by the number of publications, metric methods are the most popular methods among researchers. They postulate the existence of a certain metric in the space of objects, which helps to find anomalies. Intuitively, the anomaly has few neighbors in the instannce space, and a typical point has many. Therefore, a good measure of anomalies can be, for example, the «distance to the k-th neighbor». (See method: Local Outlier Factor). Specific metrics are used here, for example Mahalonobis distance. Mahalonobis distance is a measure of distance between vectors of random variables, generalizing the concept of Euclidean distance. Using Mahalonobis distance, it is possible to determine the similarity of unknown and known samples. It differs from Euclidean distance in that it takes into account correlations between variables and is scale invariant. alt text

    The most common form of clustering-based anomaly detection is done with prototype-based clustering.

    Using this approach to anomaly detection, a point is classified as an anomaly if its omission from the group significantly improves the prototype, then the point is classified as an anomaly. This logically makes sense. K-means is a clustering algorithm that clusters similar points. The points in any cluster are similar to the centroid of that cluster, hence why they are members of that cluster. If one point in the cluster is so far from the centroid that it pulls the centroid away from it's natural center, than that point is literally an outlier, since it lies outside the natural bounds for the cluster. Hence, its omission is a logical step to improve the accuracy of the rest of the cluster. Using this approach, the outlier score is defined as the degree to which a point doesn't belong to any cluster, or the distance it is from the centroid of the cluster. In K-means, the degree to which the removal of a point would increase the accuracy of the centroid is the difference in the SSE, or standard squared error, or the cluster with and without the point. If there is a substantial improvement in SSE after the removal of the point, that correlates to a high outlier score for that point. More specifically, when using a k-means clustering approach towards anomaly detection, the outlier score is calculated in one of two ways. The simplest is the point's distance from its closest centroid. However, this approach is not as useful when there are clusters of differing densities. To tackle that problem, the point's relative distance to it's closest centroid is used, where relative distance is defined as the ratio of the point's distance from the centroid to the median distance of all points in the cluster from the centroid. This approach to anomaly detection is sensitive to the value of k. Also, if the data is highly noisy, then that will throw off the accuracy of the initial clusters, which will decrease the accuracy of this type of anomaly detection. The time complexity of this approach is obviously dependent on the choice of clustering algorithm, but since most clustering algorithms have linear or close to linear time and space complexity, this type of anomaly detection can be highly efficient.

18. 
    How do you deal with sparse data?
    We could take a look at L1 regularization since it best fits to the sparse data and do feature selection. If linear relationship - linear regression either - svm.

    Also it would be nice to use one-hot-encoding or bag-of-words. A one hot encoding is a representation of categorical variables as binary vectors. This first requires that the categorical values be mapped to integer values. Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.


21. 
    What is the effect on the coefficients of logistic regression if two predictors are highly correlated? What are the confidence intervals of the coefficients?
    When predictor variables are correlated, the estimated regression coefficient of any one variable depends on which other predictor variables are included in the model. When predictor variables are correlated, the precision of the estimated regression coefficients decreases as more predictor variables are added to the model.

    In statistics, multicollinearity (also collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated, meaning that one can be linearly predicted from the others with a substantial degree of accuracy. In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data. Multicollinearity does not reduce the predictive power or reliability of the model as a whole, at least within the sample data set; it only affects calculations regarding individual predictors. That is, a multiple regression model with correlated predictors can indicate how well the entire bundle of predictors predicts the outcome variable, but it may not give valid results about any individual predictor, or about which predictors are redundant with respect to others.

    The consequences of multicollinearity:

    Ratings estimates remain unbiased.
    Standard coefficient errors increase.
    The calculated t-statistics are underestimated.
    Estimates become very sensitive to changes in specifications and changes in individual observations.
    The overall quality of the equation, as well as estimates of variables not related to multicollinearity, remain unaffected.
    The closer multicollinearity to perfect (strict), the more serious its consequences.
    Indicators of multicollinearity:

    High R2 and negligible odds.
    Strong pair correlation of predictors.
    Strong partial correlations of predictors.
    High VIF - variance inflation factor.
    Confidence interval (CI) is a type of interval estimate (of a population parameter) that is computed from the observed data. The confidence level is the frequency (i.e., the proportion) of possible confidence intervals that contain the true value of their corresponding parameter. In other words, if confidence intervals are constructed using a given confidence level in an infinite number of independent experiments, the proportion of those intervals that contain the true value of the parameter will match the confidence level.

    Confidence intervals consist of a range of values (interval) that act as good estimates of the unknown population parameter. However, the interval computed from a particular sample does not necessarily include the true value of the parameter. Since the observed data are random samples from the true population, the confidence interval obtained from the data is also random. If a corresponding hypothesis test is performed, the confidence level is the complement of the level of significance, i.e. a 95% confidence interval reflects a significance level of 0.05. If it is hypothesized that a true parameter value is 0 but the 95% confidence interval does not contain 0, then the estimate is significantly different from zero at the 5% significance level.

    The desired level of confidence is set by the researcher (not determined by data). Most commonly, the 95% confidence level is used. However, other confidence levels can be used, for example, 90% and 99%.

    Factors affecting the width of the confidence interval include the size of the sample, the confidence level, and the variability in the sample. A larger sample size normally will lead to a better estimate of the population parameter. A Confidence Interval is a range of values we are fairly sure our true value lies in.

    X ± Z*s/√(n), X is the mean, Z is the chosen Z-value from the table, s is the standard deviation, n is the number of samples. The value after the ± is called the margin of error.