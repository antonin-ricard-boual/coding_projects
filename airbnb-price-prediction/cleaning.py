
import numpy as np
import pandas as pd

def load_and_clean_data():

    def read_and_info(filepath):
        df = pd.read_csv(filepath)
        return df

    # Calling the function for the different datasets
    listings = read_and_info("data/listings.csv")
    locations = read_and_info("data/locations.csv")
    reviews = read_and_info("data/reviews.csv")

    desired_dtypes = {"id": "int64", "name": "string", "host_name": "string", 
                    "room_type": "string", "price": "float64", "neighbourhood": "string", "last_review": "datetime64[ns]"}

    # Looped conversion of columns
    for df in [listings, locations, reviews]:
        for col in df.columns:
            if col in desired_dtypes.keys():
                dtype = desired_dtypes[col]
                if "datetime" in dtype:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(dtype, errors="ignore")


    # This gives us the number of nas per column, for host_name because of the type string it doesn't identify for this reason I replace it by np.nan
    for df in [listings, locations, reviews]:
        for col in df.columns:
            df[col] = df[col].replace("nan", np.nan)
            df[col] = df[col].replace("<NA>", np.nan)


    # For the listings df: Price will serve as a target for our machine learning part, I will 
    # therefore delete all rows that do not contain a price. I will also convert all prices that 
    # are negative to positive prices.
    listings = listings.dropna(subset=["price"])
    listings["price"] = listings["price"].abs()

    # When we don't know the name of the host, I will replace it with, "notknown", we want to keep this 
    # column, as otherwise we lose many observations and the name of the host is not very important
    listings["host_name"] = listings["host_name"].replace(np.nan, "notknown")

    # For the reviews df: Where the reviews per month are not listed I will replace them with the value 0
    reviews["reviews_per_month"] = reviews["reviews_per_month"].replace(np.nan, 0)

    # Some of the values of last review have an extra 100 years added to them, I substract these 100 years 
    # to get the correct date, we leave last review columnn as is otherwise, we will drop it after anyway
    mask = reviews["last_review"].dt.year > 2050
    reviews.loc[mask, "last_review"] = reviews.loc[mask, "last_review"] - pd.DateOffset(years=100)

    master_first = listings.merge(locations, on="id", how="inner")
    master = master_first.merge(reviews, on = "id", how = "inner")
    master.shape
    master.head()
    return master
