# INFO 370 Final Resource

Calvin Chan, Alex Huang, Kevin Huynh, Rachel Vuu

## Project Description

For our final project, we selected the [Seattle Airbnb dataset](https://www.kaggle.com/airbnb/seattle), which contains a calendar file documenting the price point and availability of a listing on a day of the year, a descriptive file with a number of features including neighborhood, price, and ratings, and a file consisting of guest reviews. 

Airbnb is one of the many technology companies that have completely transformed the sharing economy. Paving the way for other crowd-based marketplaces, Airbnb is one of several prominent startups collectively disrupted the business models of traditional industries (Satopaa & Mehrotra, 2018). Airbnb affords a platform for prospective guests to book accommodations with more flexibility and potentially lower price points, and for hosts to list their properties.

For the scope of this project, the audience we are targeting is potential Airbnb hosts who are considering investing in or renting out a property. For people who need extra income, Airbnb offers a self-regulating (via guest and host reviews), free-market solution for collecting short-term market-rate rents. However, inexperienced and longtime hosts may not be getting the most out of their listing and are missing out on potential revenue every rent day.

In an article by the New York Times, a host says that she grosses nearly $100,000 each year through Airbnb. Despite this income, she still does not feel financially secure due to the costs of upkeep as well as the “slower” months of January, February, and March (Dobbins, 2017). Depending on the location, property size, and other socio-economic factors, hosts may benefit more or less. Additionally, it has been found that Airbnb hosts use marketing logic to target specific demographics  (Lutz & Newlands, 2017). A weak alignment between consumer segmentation and this host targeting can lead to potentially reduced matching efficiency. By performing data analyses on the features of the Seattle Airbnb dataset, we can draw insights on whether properties with a specific set of attributes bring more financial gain than others. 

Through our resource, users will gain greater insight into what factors impact the price of a listing. Since there are so many different attributes for a listing, hosts will come with a better understanding of competitive prices. Hosts, interested with garnering high reviews, also will learn what factors influence ratings and reviews. This will allow hosts to stand out among the many listings, attracting more customers and making more money while maximizing the potential of their listing.

**What statistical and machine learning methods do you plan on using to test your hypothesis?**

Hypotheses we are testing include...
- The greater the number of amenities, the higher the price of the listing.
- The closer the location of the listing to downtown Seattle, the higher the price of the listing.
- The better customer service provided by the host, the higher the review ratings of the listing.

## Technical Description

The format of our final web resource will be an HTML page. Building an informative web application that displays aesthetic, interactive visualizations is our goal.

Some of the data collection and management challenges that we anticipate include data wrangling and feature generation. Since there are at least three different .csv files, we will need to cleanly join the datasets together so that dates, prices, reviews, and more are correctly matched to each listing. There are many categorical variables within the datasets, so it will be important to use feature generation to create dummy variables in order to perform statistical and machine learning models.

The technical skills that we will need to learn in order to complete the project is more on the use of unsupervised learning. With our goal of determining the optimal features that has the largest effect on the price and valuation of an Airbnb, unsupervised learning could potentially offer us the insight of which of the many features has the most significant impact.

We will be performing data prepping at first, handling the few missing data values by using the average of the values of that feature. If there are entire columns that lack sufficient information, we would take those data rows out of our analysis. Next we would be conducting exploratory data analysis that would lead us to feature engineering, and selecting key features that would eventually serve as the ones we will be using in our model. We will be conducting our data modeling by comparing the effectiveness of different models on our dataset, testing out for the highest score that each model could provide us with based on the parameters we decide to give the models. We will begin by using the sklearn api to the preprocess the data into a form where we can pass it into the model selection modules such as grid search and cross validation. We will pick specific features based on their correlation on our responding variable, number of listings taken, and pick the features with the strongest correlations and removing the rest. We will start out this method and compare the results, grid search score, with the OLS/GLM method to see which result brings us closer to the best result possible. 

Some of the major challenges we anticipate involves dealing with free-form text in certain columns for our datasets. In the past, we have worked with categorical variables with a fixed set of answers. In these Airbnb datasets, columns such as 'neighborhood_overview', 'notes', 'host_about', and more have unique values. We will have to decide if and how we want to include these variables in our models.

## References

Dobbins, J. (2017, April 07). Making a Living With Airbnb. Retrieved from https://www.nytimes.com/2017/04/07/realestate/making-a-living-with-airbnb.html

Lutz, C., & Newlands, G. (2017, June 3). Consumer segmentation within the sharing economy: The case of Airbnb. Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0148296318301474

Satopaa, V., & Mehrotra, P. (2018, March 15). Disrupting business models is not enough. We need tech innovation too. Retrieved from https://www.weforum.org/agenda/2018/03/sharing-economy-product-innovation-balance-disruption/
