# Predictive Maintenance on a Jet Engine

Can We predict when a part will fail?

* I want to be able to predict how long till an engine will fail based on sensor readings.

You can find the the dataset I used for this project at [**this link**](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). Of the 4 potential datasets, I used the FD003 dataset. This data is collected from 100 different engines. These engines are run till failure is hit. Sensor readings are recorded once during each cycle. A cycle is defined as one instance of engine start to engine shutdown. This is more or less equivalent to number of trips the engine has been on.

![max cycle](img/max_cycle_hist.png)

The above graph shows the distribution of the life of the engines.
## Data Cleaning

The data did not need much cleaning. All columns were already numeric. I did have to remove some columns. I removed all columns that did not contain unique data within the column. I also removed any columns that had no effect on the model.
The columns were not defined within the CSVs. I used a dictionary in the variables file to define column names. I also created a column for Remaining Useful Life (RUL). This was done by subtracting the the current cycle by the maximum cycle amount for each engine.

## Model Creation and Testing

I chose to use a Random Forest Regressor. I tested 4 other models. I used R^2 and RMSE to evaluate the models. My intial testing was showing very bad scores across the board. The engines with very long lifespans skewed the data. To correct this, I used the clip method in pandas to limit the maximum RUL to 120. I want to focus on the data as it gets closer to 0 RUL. This transformation allowed me to start seeing accurate predictions from the model. None of the other models came close to matching the scores I was getting with the Random Forest Regerssor. I ran several itterations of Gridsearch CV, but none of the changes resulted in a significant improvement in R^2 or RMSE.

## Results

I found that the biggest factor in predicting engine health was sensor 11 (HPC Outlet Static Pressure). 

![feature importance](img/feature_importance.png)

### Notes
Topic: Predictive maintenance is projecting when a part or system is going to fail. Using those projections, we can schedule maintenance in advance to prevent in flight problems and limit costly damage to equipment.

Data: I got my data from a study done by NASA on 100 turbo fan jet engines. They recorded 21 sensors through their life cycle till failure. Each cycle is approximately one flight. This graph shows the distribution of the life span of each engine tested. We can see from this that the normal lifespan for this particular engine is around 150 - 300 cycles. 

Jet Engines: They are very similar to combustions engines in cars, once broken down. You intake air, it gets compressed, the compressed air is mixed with fuel, the air/fuel mixture is ignited, and the resulting combustion is used for thrust.

Indicators: The first that we need to know is what sensors are telling us the engine will fail. Biggest indicator was Hight Pressure Compressor (HPC) Static Pressure. HPC static pressure is measured after the HPC and before the Combustor. There are several different factors that can cause this.  

Test Engine Comparison: These could be further expanded to be able to see what parts are failing based on the individual sensor data.
