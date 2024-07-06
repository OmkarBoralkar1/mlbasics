#
# English:
#
# Bias and Variance:
# Bias and variance are components of the predictive error in machine learning models.
#
# Bias: It represents the error introduced by approximating a real-world problem, and it leads to underfitting.
# High bias implies the model is too simplistic and does not capture the underlying patterns in the data.
#
# Variance: It represents the model's sensitivity to fluctuations in the training data, leading to overfitting.
# High variance suggests the model is too complex and captures noise in the training data.
#
# Types of Variance:
#
# High Variance: The model is too sensitive to training data and captures noise, leading to overfitting.
#
# Low Variance: The model is less sensitive to training data, but it might oversimplify the underlying patterns,
# leading to underfitting.
#
# Types of Bias:
#
# High Bias: The model is too simplistic and misses the underlying patterns, leading to underfitting.
#
# Low Bias: The model captures the underlying patterns well, but it might be too sensitive to fluctuations,
# leading to overfitting.
#
# No Bias: The ideal state where the model perfectly captures the underlying patterns without underfitting or
# overfitting.
#
# Overfitting, Underfitting, and Balanced Fit:
#
# Overfitting: The model learns the training data too well, capturing noise and performing poorly on new, unseen data.
#
# Underfitting: The model is too simplistic, failing to capture the underlying patterns in the training data
# and performing poorly on both training and new data.
#
# Balanced Fit: The model strikes a balance, capturing underlying patterns without being too sensitive to
# noise or fluctuations, leading to good performance on both training and new data.
#
# Example:
# Consider a polynomial regression:
#
# High Bias (Underfitting): Using a linear model to fit a quadratic dataset, missing the curvature.
#
# Low Bias, High Variance (Overfitting): Using a high-degree polynomial to fit a dataset, capturing noise and fluctuations.
#
# Balanced Fit: Using an appropriately chosen polynomial degree that captures the underlying pattern
# without overfitting or underfitting.


# Real-Life Example:
# Consider a student preparing for exams.
#
# High Bias (Underfitting): The student studies only one topic and fails to cover the broader syllabus.
# Low Bias: The student thoroughly covers a diverse range of topics, understanding the complexities of the broader syllabus.
# High Variance (Overfitting): The student memorizes every question from a set of practice exams without understanding the concepts.
# Low Variance: The student avoids memorization but focuses on grasping underlying concepts, making sure
# not to overemphasize specific practice questions.
# Balanced Fit: The student focuses on understanding core concepts, practices a diverse set of problems,
# and performs well on both practice exams and the actual test.
# Balanced Fit (Low Bias and Low Variance): The student strikes a balance by exploring various subjects,
# understanding core concepts, and practicing a mix of problems. This approach leads to a well-rounded preparation,
# resulting in good performance on both practice exams and the actual test.
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#
# Hinglish:
#
# Bias aur Variance:
# Bias aur variance machine learning models mein predictive error ke components hote hain.
#
# Bias: Ye real-world problem ko approximate karne mein aane wala error represent karta hai,
# aur isse underfitting hota hai. High bias ka matlab hai ki model bahut simple hai aur data ke
# underlying patterns ko capture nahi kar pata.
#
# Variance: Ye model ki sensitivity ko training data ke fluctuations ke liye represent karta hai,
# jo ki overfitting ka karan ho sakta hai. High variance ka matlab hai ki model bahut complex
# hai aur training data ke noise ko capture kar leta hai.
#
# Types of Variance:
#
# High Variance: Model training data ke fluctuations ke liye bahut sensitive hai aur noise ko capture kar leta hai,
# jisse overfitting hota hai.
#
# Low Variance: Model training data ke liye kam sensitive hai, lekin ye underlying patterns ko oversimplify kar sakta hai,
# jisse underfitting hota hai.
#
# Types of Bias:
#
# High Bias: Model bahut simple hai aur underlying patterns ko miss kar deta hai, jisse underfitting hota hai.
#
# Low Bias: Model underlying patterns ko acche se capture karta hai, lekin ye fluctuations ke liye bahut sensitive
# ho sakta hai, jisse overfitting hota hai.
#
# No Bias: Ye ideal state hai jisme model underlying patterns ko perfect taur par capture karta hai
# bina underfitting ya overfitting ke.
#
# Overfitting, Underfitting, aur Balanced Fit:
#
# Overfitting: Model training data ko bahut acche se sikhta hai, noise ko bhi capture karta hai,
# lekin new, unseen data par accha perform nahi karta.
#
# Underfitting: Model bahut simple hai, underlying patterns ko capture nahi kar pata, aur iska performance
# both training aur new data par bhi kam hota hai.
#
# Balanced Fit: Model ek balance maintain karta hai, underlying patterns ko capture karta hai bina noise ya
# fluctuations ke bahut sensitive hone ka risk liye, jisse ki ye both training aur new data par accha perform kare.
#
# Example:
# Ek polynomial regression ko consider karein:
#
# High Bias (Underfitting): Ek quadratic dataset ko fit karne ke liye linear model ka istemal karna,
# jisme curvature miss hota hai.
#
# Low Bias, High Variance (Overfitting): Dataset ko fit karne ke liye high-degree polynomial ka istemal karna,
# jisme noise aur fluctuations ko capture kiya jata hai.
#
# Balanced Fit: Sahi tarah se chuna gaya polynomial degree ka istemal karna, jo underlying pattern ko
# capture karta hai bina overfitting ya underfitting ke.
#

# Real-Life Example:
#
# Consider karo ek student ko jo exams ke liye tayari kar raha hai:
#
# High Bias (Underfitting): Student sirf ek topic par focus karta hai aur broader syllabus ko ignore karta hai.
#
# Low Bias: Student thorough taur par ek diverse range ke topics ko cover karta hai, broader syllabus ke complexities ko samajhte hue.
#
# High Variance (Overfitting): Student har question ko yaad kar leta hai ek set ke practice exams se bina concepts ko samjhe.
#
# Low Variance: Student yaad karna se bachta hai, aur underlying concepts ko samajhne par dhyan deta hai,
# dhyan rakh kar ki specific practice questions par zyada emphasis na ho.
#
# Balanced Fit: Student core concepts ko samajhne par focus karta hai, ek diverse set ke problems ko practice karta hai,
# aur dono practice exams aur actual test mein acche se perform karta hai.
#
# Balanced Fit (Low Bias aur Low Variance): Student ek balance banata hai alag-alag subjects ko explore karke,
# core concepts ko samajhkar, aur ek mix of problems ko practice karke. Ye approach ek acchi tayari ka marg darshata hai,
# jisse ki dono practice exams aur actual test mein accha perform kiya ja sake.