# Project Description

For our final project, we selected the Seattle Airbnb dataset, which contains a calendar file documenting the price point and availability of a listing on a day of the year, a descriptive file with a number of features including neighborhood, price, and ratings, and a file consisting of guest reviews. 

Airbnb is one of the many technology companies that have completely transformed the sharing economy. Paving the way for other crowd-based marketplaces, Airbnb is one of several prominent startups collectively disrupted the business models of traditional industries (Satopaa & Mehrotra, 2018). Airbnb affords a platform for prospective guests to book accomodations with more flexibility and potentially lower price points, and for hosts to list their properties. For the scope of this project, the audience we are targeting is potential Airbnb hosts who are considering investing in a property. For people who need extra income, Airbnb offers a self-regulating (via guest and host reviews), free-market solution for collecting short-term market-rate rents. 

In an article by the New York Times, a host says that she grosses nearly $100,000 each year through Airbnb. Despite this income, she still does not feel financially secure due to the costs of upkeep as well as the “slower” months of January, February, and March (Dobbins, 2017). Depending on the location, property size, and other socio-economic factors, hosts may benefit more or less. Additionally, it has been found that Airbnb hosts use marketing logic to target specific demographics  (Lutz & Newlands, 2017). A weak alignment between consumer segmentation and this host targeting can lead to potentially reduced matching efficiency. By performing data analyses on the features of the Seattle Airbnb dataset, we can draw insights on whether properties with a specific set of attributes bring more financial gain than others. 

Through our resource, users will gain greater insight into what factors impact the price of a listing. Since there are so many different attributes for a listing, hosts will come with a better understanding of competitive prices. Hosts, interested with garnering high reviews, also will learn what factors influence ratings and reviews. This will allow hosts to stand out among the many listings, attracting more customers and making more money while maximizing the potential of their listing.

### Hypothesis:
AirBnB listing will spike around during the holiday seasons in the Seattle area due to people returning home or visiting family. 
AirBnB listing will increase during the summer and winter months due to the sunny weather and the snowy seasons.
During times with special events like concerts, conventions, and meetups, the listings within the center of Seattle or near downtown Seattle will increase in the amount of listings and fillings. 
References:
Dobbins, J. (2017, April 07). Making a Living With Airbnb. Retrieved from https://www.nytimes.com/2017/04/07/realestate/making-a-living-with-airbnb.html
Lutz, C., & Newlands, G. (2017, June 3). Consumer segmentation within the sharing economy: The case of Airbnb. Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0148296318301474 
Satopaa, V., & Mehrotra, P. (2018, March 15). Disrupting business models is not enough. We need tech innovation too. Retrieved from https://www.weforum.org/agenda/2018/03/sharing-economy-product-innovation-balance-disruption/

## Technical Description

This section of your proposal is an opportunity to think through the specific analytical steps you'll need to complete throughout the project.

What will be the format of your final web resource (Shiny app, HTML page or slideshow compiled with KnitR, etc.)?

    - The format of our final product will be an HTML page.
    
Do you anticipate any specific data collection / data management challenges?

    - Since there are at least three different .csv files, we will need to cleanly join the datasets together so that dates, prices, reviews, and more are correctly matched to each listing. 

    - There are many categorical variables within the datasets, so it will be important to use feature generation to create dummy variables in order to perform statistical and machine learning models.

What new technical skills will need to learn in order to complete your project?

    - The technical skills that we will need to learn in order to complete the project is more on the use of unsupervised learning. With our goal of determining the optimal features that has the largest effect on the price and valuation of an Airbnb, unsupervised learning could potentially offer us the insight of which of the many features has the most significant impact. 
    
How will you conduct you analysis? Please include a detailed description of your intended modeling approach.

    - We will be performing data prepping at first, handling the few missing data values by using the average of the values of that feature. If there are entire columns that lack sufficient information, we would take those data rows out of our analysis. Next we would be conducting exploratory data analysis that would lead us to feature engineering, and selecting key features that would eventually serve as the ones we will be using in our model. We will be conducting our data modeling by comparing the effectiveness of different models on our dataset, testing out for the highest score that each model could provide us with based on the parameters we decide to give the models. We will begin by using the sklearn api to the preprocess the data into a form where we can pass it into the model selection modules such as grid search and cross validation. We will pick specific features based on their correlation on our responding variable, number of listings taken, and pick the features with the strongest correlations and removing the rest. We will start out this method and compare the results, grid search score, with the OLS/GLM method to see which result brings us closer to the best result possible. 
    
What major challenges do you anticipate? 

    - One major challenge we anticipate is the understanding the weighted scale of different features in different cities. The valuation of an Airbnb and the predicted profitability is determined by a variety of combinations of features. Every city has its own set of valuable features that people are interested in. For instance, people who are looking for Airbnb’s in Southern California would be interested in the proximity of the house to the beach, while features that people would be interested in for locations in New York would be the proximity to Times Square and the main city attractions. Based on cities with different sets of interests and priorities, we would need to come up with a different set of features that are optimal in that particular location.

