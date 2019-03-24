# coding=utf-8
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from scipy import stats


class CountryCentroid:
    """Class for finding the location of distribution (convex hull peeling depth)
    of a given county by means of recursive convex hull algorithm.
    More on : https://en.wikipedia.org/wiki/Convex_layers

    Constructor arguments:
    :param csv_file: data set containing the data for each location
    :param cntry_id: country ID in ISO 3166-1 alpha-2 format
    :param min_pop: minimum population of the towns that form the layer vertices
    :param _3d: If True, the country will be plotted in cartesian
    """

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
        self.layers = []

    def load_df(self):
        """
        loads the data set as a dataframe
        :return: Pandas dataframe
        """
        return pd.read_csv(
            self.csv,
            low_memory=False,
            index_col=False,
            encoding='utf-8'
        )

    def set_country(self, country_id):
        """
        Sets the country that you want to find its central tendency town
        :param country_id: country ID in ISO 3166-1 alpha-2 format
        :return: None
        """
        cntry_df = self.df[
            self.df['Country'] == country_id
            ]

        setattr(self, 'df', cntry_df)

    def set_min_population(self, min_pop):
        """
        Sets the towns minimum population that you want to have as vertices
        :param min_pop: minimum population
        :return: None
        """
        pop_df = self.df[
            (self.df['Population'] >= min_pop) &
            pd.notnull(self.df['Population'])
            ]

        setattr(self, 'df', pop_df)

    def project_to_equirectangular(self):
        """
        Projects the geographic coordinate system on an equirectangular plane
        X = R * ϕ * cos(ψ0)
        Y = R * ψ
        :return: The numpy array of X and Y on equirectangular plane
        """
        earth_r = 6371
        lat_mean = self.df['Latitude'].mean()

        self.df['X'] = self.df.apply(
            # X = R * ϕ * cos(ψ0)
            lambda row: earth_r * row['Longitude'] * np.cos(lat_mean),
            axis=1
        )

        self.df['Y'] = self.df.apply(
            # Y = R * ψ
            lambda row: earth_r * row['Latitude'],
            axis=1
        )
        return self.df[['X', 'Y']].values

    def convert_deg_to_radian(self):
        """
        Converts the geographic coordinates from degrees to radians
        :return: None
        """
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
        Projects the geographic coordinates on a cartesian system
        X = R * cos(lat) * cos(lon)
        Y = R * cos(lat) * sin(lon)
        Z = R * sin(lat)
        :return: The numpy array of X, Y and Z on cartesian coordinate system
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
        """
        Removes the outliers by means of standard score method
        :return: None
        """
        self.df = self.df[
            (np.abs(
                stats.zscore(self.df[['Longitude', 'Latitude']])
            ) < 3).all(axis=1)
        ]

    def plot_country(self, mode='2d'):
        """
        Plots the country towns
        :param mode: projection either in 3d or 2d
        :return: None
        """
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
        """
        Plots the towns in a scatter diagram
        :return: None
        """
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')

    def make_convex_hull(self):
        """
        Instantiates the ConvexHull class
        :return: ConvexHull object
        """
        return ConvexHull(self.points)

    def find_vertices(self):
        """
        Sets the convex vertices for the remaining points(towns)
        :return: None
        """
        setattr(self, 'vertices', self.make_convex_hull().vertices)

    def plot_convex_layer(self):
        """
        Plots the outer convex layer
        :return: None
        """
        plt.plot(
            self.points[self.vertices, 0],
            self.points[self.vertices, 1],
            'r--',
            lw=1
        )
        plt.plot(
            self.points[self.vertices[0], 0],
            self.points[self.vertices[0], 1],
            'ro'
        )

    def peel_outer_convex_layer(self):
        """
        Removes the outer convex layer vertices and appends them to layers attr
        :return: None
        """
        self.layers.append(self.points[self.vertices.tolist(), :])
        self.points = np.delete(self.points, self.vertices.tolist(), 0)
        self.find_vertices()

    def peel(self, plot=True):
        """
        Removes the convex layers recursively until the central tendency town is found
        :param plot: if True, plots the layers
        :return: the remaining points at the end of "Convex Hull Peeling Depth"
        """
        if plot:
            self.plot_convex_layer()
        while self.points.shape[0] > 2:
            try:
                self.peel_outer_convex_layer()
                if plot:
                    self.plot_convex_layer()
            except Exception:
                break

        return self.points

    def find_center_of_dist(self):
        """
        Finds the most populous centroid
        :return: dataframe containing the center of distribution data
        """
        pnts = self.peel(plot=False)
        if len(pnts) == 1:
            return self.df[
                (self.df['X'] == pnts[0][0]) &
                (self.df['Y'] == pnts[0][1])
                ]
        elif len(pnts) == 0:
            self.df = self.df[
                (self.df['X'].isin(self.layers[-1][:][:, 0])) &
                (self.df['Y'].isin(self.layers[-1][:][:, 1]))
                ]
            return self.df.loc[
                   self.df['Population'].idxmax(), :
                   ]
        else:
            self.df = self.df[
                (self.df['X'].isin(pnts[:][:, 0])) &
                (self.df['Y'].isin(pnts[:][:, 1]))
                ]
            return self.df.loc[self.df['Population'].idxmax(), :]


if __name__ == '__main__':
    # Example:
    # Find the centroid town in Italy with a minimum population of 1000
    cc = CountryCentroid('world-cities-database.zip', cntry_id='it', min_pop=1000)
    # remove the outlier towns
    cc.remove_outliers_by_zscore()
    # plot the towns on a scatter diagram
    cc.scatter_towns()
    # find the convex vertices (first layer)
    cc.find_vertices()
    # peel the convex layers recursively
    cc.peel()
    # Show on 2d plot
    plt.show()
    # find the most populous town as the center of distribution
    cc.find_center_of_dist()

