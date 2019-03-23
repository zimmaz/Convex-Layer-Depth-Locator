import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats


class CountryCentroid:
    def __init__(self, csv_file, cntry_id, min_pop=0, _3d=False):
        self.csv = csv_file
        self.df = self.load_df()
        self.set_country(cntry_id)
        self.set_min_population(min_pop=min_pop)
        if not _3d:
            self.points = self.project_to_equirectangular()
        else:
            self.points = self.project_to_cartesian()
        self.vertices = None

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
        cntry_df = self.df[
            self.df['Country'] == country_id
            ]

        setattr(self, 'df', cntry_df)

    def set_min_population(self, min_pop):
        pop_df = self.df[
            (self.df['Population'] >= min_pop) &
            pd.notnull(self.df['Population'])
            ]

        setattr(self, 'df', pop_df)

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

    def convert_deg_to_radian(self):
        self.df['Longitude'] = self.df.apply(
            lambda row: np.radians(row['Longitude']),
            axis=1
        )
        self.df['Latitude'] = self.df.apply(
            lambda row: np.radians(row['Latitude']),
            axis=1
        )

    def project_to_cartesian(self):
        """
        x = R * cos(lat) * cos(lon)
        y = R * cos(lat) * sin(lon)
        z = R *sin(lat)
        :param df: dataframe
        :return:
        """
        earth_r = 6371
        self.convert_deg_to_radian()

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

    def remove_outliers_by_zscore(self):
        self.df = self.df[
            (np.abs(
                stats.zscore(self.df[['Longitude', 'Latitude']])
            ) < 3).all(axis=1)
        ]

    def plot_country(self, mode='2d'):
        if mode == '2d':
            cntry = self.project_to_equirectangular()
            plt.plot(
                cntry[:, 0],
                cntry[:, 1],
                'o'
            )
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cntry = self.project_to_cartesian()
            ax.scatter(
                cntry[:, 0],
                cntry[:, 1],
                cntry[:, 2],
                c=cntry[:, 2],
                linewidth=1
            )

    def scatter_towns(self):
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')


    def make_convex_hull(self):
        return ConvexHull(self.points)

    def set_vertices(self):
        setattr(self, 'vertices', self.make_convex_hull().vertices)

    def plot_convex_layer(self):
        plt.plot(self.points[self.vertices, 0], self.points[self.vertices, 1], 'r--', lw=1)

    def peel_outer_convex_layer(self):
        self.points = np.delete(self.points, self.vertices.tolist(), 0)
        self.set_vertices()

    def peel(self):
        self.plot_convex_layer()
        while self.points.shape[0] > 2:
            try:
                self.peel_outer_convex_layer()
                self.plot_convex_layer()
            except ValueError:
                break


if __name__ == '__main__':
    cc = CountryCentroid('world-cities-database.zip', cntry_id='ir', min_pop=10)
    cc.remove_outliers_by_zscore()
    cc.scatter_towns()
    cc.set_vertices()
    cc.peel()
    plt.show()
