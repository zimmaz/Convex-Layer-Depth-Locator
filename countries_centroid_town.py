import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


class CountryCentroid:
    def __init__(self, csv_file):
        self.csv = csv_file
        self.df = self.load_df()

    def load_df(self):
        return pd.read_csv(
            self.csv,
            low_memory=False,
            index_col=False,
            encoding='utf-8'
        )

    @staticmethod
    def load_pickle_dataframe(df):
        return df.pd.read_pickle('points.pkl')

    def set_country(self, country_id):
        return self.df[
            (self.df['Country'] == country_id) &
            pd.notnull(self.df['Population'])
            ][['Latitude', 'Longitude']]

    def project_to_equirectangular(self):
        """
        x=R * ϕcos(ψ0)
        y=R * ψ
        :param df:
        :return:
        """
        earth_r = 6371
        lat_mean = self.df['Latitude'].mean()

        self.df['X'] = self.df.apply(
            lambda row: earth_r * row['Longitude'] * np.cos(lat_mean),
            axis=1
        )

        self.df['Y'] = self.df.apply(
            lambda row: earth_r * row['Latitude'],
            axis=1
        )
        return self.df[['X', 'Y']].values

    def project_to_cartesian(self):
        """
        x = R * cos(lat) * cos(lon)
        y = R * cos(lat) * sin(lon)
        z = R *sin(lat)
        :param df: dataframe
        :return:
        """
        earth_r = 6371

        self.df['X'] = self.df.apply(
            lambda row: earth_r * np.cos(row['Latitude']) * np.cos(row['Longitude']),
            axis=1
        )

        self.df['Y'] = self.df.apply(
            lambda row: earth_r * np.cos(row['Latitude']) * np.sin(row['Longitude']),
            axis=1
        )

        self.df['Z'] = self.df.apply(
            lambda row: earth_r * np.sin(row['Latitude']),
            axis=1
        )
        return self.df[['X', 'Y', 'Z']].values

    def make_convex_hull(self, mode='2d'):
        if mode == '2d':
            return ConvexHull(self.df[['X', 'Y']].values)
        else:
            return ConvexHull(self.df[['X', 'Y', 'Z']].values)


if __name__ == '__main__':
    cc = CountryCentroid('world-cities-database.zip')
    x, y, z = cc.project_to_cartesian()
    ax.scatter(x, y, z, c=y, linewidth=1)
    plt.show()
