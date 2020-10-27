import pandas as pd
import numpy as np


class ETL:
    """
    Class ETL to handle the ETL of the data.

    This class really only does the extract and transform functions of ETL. The data is then received downstream by the
        algorithms for processing.
    """
    def __init__(self, data_name, random_state=1):
        """
        Init function. Takes a data_name and extracts the data and then transforms.

        All data comes from the data folder. The init function calls to both extract and transform for processing

        :param data_name: str, name of the data file passed at the command line. Below are the valid names:
            breast-cancer
            glass
            soybean
            abalone
            forest-fires
            machine
        :param random_state: int, seed for data split
        """
        # Set the attributes to hold our data
        self.data = None
        self.transformed_data = None
        self.tune_data = None
        self.test_split = {}
        self.train_split = {}

        # Meta attributes
        self.data_name = data_name
        self.random_state = random_state
        self.classes = 0
        self.class_names = None

        # Extract
        self.extract()

        # Transform
        self.transform()

        # Split
        if self.classes > 0:
            self.cv_split_classification()
        else:
            self.cv_split_regression()

        # Combine Train Sets
        self.cv_combine()

    def extract(self):
        """
        Function to extract data based on data_name passed

        :return self.data: DataFrame, untransformed data set
        """
        # breast-cancer
        if self.data_name == 'breast-cancer':
            column_names = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                            'Normal_Nucleoli', 'Mitoses', 'Class']
            self.data = pd.read_csv('data\\breast-cancer-wisconsin.data', names=column_names)

        # glass
        elif self.data_name == 'glass':
            column_names = ['ID', 'Refractive_Index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium',
                            'Calcium', 'Barium', 'Iron', 'Class']
            self.data = pd.read_csv('data\\glass.data', names=column_names)

        # soybean
        elif self.data_name == 'soybean':
            column_names = ['Date', 'Plant_Stand', 'Percip', 'Temp', 'Hail', 'Crop_Hist', 'Area_Damaged', 'Severity',
                            'Seed_Tmt', 'Germination', 'Plant_Growth', 'Leaves', 'Leaf_Spots_Halo', 'Leaf_Spots_Marg',
                            'Leaf_Spot_Size', 'Leaf_Shread', 'Leaf_Malf', 'Leaf_Mild', 'Stem', 'Lodging',
                            'Stem_Cankers', 'Canker_Lesion', 'Fruiting_Bodies', 'External_Decay', 'Mycelium',
                            'Int_Discolor', 'Sclerotia', 'Fruit_Pods', 'Fruit_Spots', 'Seed', 'Mold_Growth',
                            'Seed_Discolor', 'Seed_Size', 'Shriveling', 'Roots', 'Class']
            self.data = pd.read_csv('data\\soybean-small.data', names=column_names)

        # abalone
        elif self.data_name == 'abalone':
            column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_Weight', 'Shucked_Weight', 'Viscera_Weight',
                            'Shell_Weight', 'Rings']
            self.data = pd.read_csv('data\\abalone.data', names=column_names)

        # machine
        elif self.data_name == 'machine':
            column_names = ['Vendor', 'Model_Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
            self.data = pd.read_csv('data\\machine.data', names=column_names)

        # forest-fires
        elif self.data_name == 'forest-fires':
            self.data = pd.read_csv('data\\forestfires.data')

        # If an incorrect data_name was specified we'll raise an error here
        else:
            raise NameError('Please specify a predefined name for one of the 6 data sets (breast-cancer, car,'
                            'segmentation, abalone, machine, forest-fires)')

    def transform(self):
        """
        Function to transform the specified data

        This is a manager function that calls to the actual helper transform function.
        """
        # breast-cancer
        if self.data_name == 'breast-cancer':
            self.transform_breast_cancer()

        # glass
        elif self.data_name == 'glass':
            self.transform_glass()

        # soybean
        elif self.data_name == 'soybean':
            self.transform_soybean()

        # abalone
        elif self.data_name == 'abalone':
            self.transform_abalone()

        # machine
        elif self.data_name == 'machine':
            self.transform_machine()

        # forest-fires
        elif self.data_name == 'forest-fires':
            self.transform_forest_fires()
        # The extract function should catch this but lets throw again in case
        else:
            raise NameError('Please specify a predefined name for one of the 6 data sets (glass, segmentation, vote,'
                            'abalone, machine, forest-fires)')

    def transform_breast_cancer(self):
        """
        Function to transform breast-cancer data set

        For this function missing data points are removed and data is normalized around 0

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # Remove missing data points
        self.data = self.data.loc[self.data['Bare_Nuclei'] != '?']
        self.data['Bare_Nuclei'] = self.data['Bare_Nuclei'].astype(int)
        self.data.reset_index(inplace=True, drop=True)

        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # We don't need ID so let's drop that
        temp_df.drop(columns='ID', inplace=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Set the class back, the normalize above would have normalized the class as well
        normalized_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object
        self.classes = 2
        self.transformed_data = normalized_temp_df

        # Class
        self.class_names = temp_df['Class'].unique().tolist()

    def transform_glass(self):
        """
        Function to transform glass data set

        For this function data is normalized around 0

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # We don't need ID so let's drop that
        temp_df.drop(columns='ID', inplace=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Set the class back, the normalize above would have normalized the class as well
        normalized_temp_df['Class'] = temp_df['Class']
        normalized_temp_df['Class'] = 'C' + normalized_temp_df['Class'].astype(str)

        # Set attributes for ETL object
        self.classes = 6
        self.transformed_data = normalized_temp_df

        # Class
        self.class_names = normalized_temp_df['Class'].unique().tolist()

    def transform_soybean(self):
        """
        Function to transform soybean data set

        For this function data is normalized around 0

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Set the class back
        normalized_temp_df.drop(columns=['Class', 'Fruit_Spots', 'Leaf_Malf', 'Leaf_Mild', 'Leaf_Shread',
                                         'Leaf_Spot_Size', 'Leaf_Spots_Halo', 'Leaf_Spots_Marg', 'Mold_Growth',
                                         'Plant_Growth', 'Seed', 'Seed_Discolor', 'Seed_Size', 'Shriveling', 'Stem'],
                                inplace=True)
        normalized_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object
        self.classes = 4
        self.transformed_data = normalized_temp_df

        # Class
        self.class_names = temp_df['Class'].unique().tolist()

    def transform_abalone(self):
        """
        Function to transform abalone data set

        For this function all numeric data is normalized using mean/standard deviation

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Add Binned variables for sex
        normalized_temp_df = normalized_temp_df.join(pd.get_dummies(temp_df['Sex'], columns=['Sex']))

        # We'll remove the old binned variables and reorder our target
        normalized_temp_df.drop(columns=['Rings', 'Sex'], inplace=True)
        normalized_temp_df['Rings'] = temp_df['Rings']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.transformed_data = normalized_temp_df

    def transform_machine(self):
        """
        Function to transform machine data set

        For this function all numeric data is normalized using mean/standard deviation

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # We'll remove unneeded variables as well as denormalize the target
        normalized_temp_df.drop(columns=['Vendor', 'Model_Name', 'ERP'], inplace=True)
        normalized_temp_df['PRP'] = temp_df['PRP']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.transformed_data = normalized_temp_df

    def transform_forest_fires(self):
        """
        Function to transform forest-fires data set

        For this function all numeric data is normalized using mean/standard deviation

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Normalize Data
        normalized_temp_df = (temp_df - temp_df.mean()) / temp_df.std()

        # Add Binned variables for sex
        normalized_temp_df = normalized_temp_df.join(pd.get_dummies(temp_df[['month', 'day']],
                                                                    columns=['month', 'day']))

        # We'll remove the old binned variables and reorder our target
        normalized_temp_df.drop(columns=['month', 'day', 'area'], inplace=True)
        normalized_temp_df['area'] = temp_df['area']

        # Set attributes for ETL object, there are two total classes so this is a singular classifier
        self.transformed_data = normalized_temp_df

    def cv_split_classification(self):
        """
        Function to split our transformed data into 10% tune and 5 cross tune splits for classification

        First this function randomizes a number between one and 10 to split out a tune set. After a number is
            randomized and the data is sorted over the class and random number. The index of the data is then mod by 5
            and each remainder represents a set for cv splitting.

        :return self.test_split: dict (of DataFrames), dictionary with keys (tune, 0, 1, 2, 3, 4) referring to the
            split transformed data
        """
        # Define base data size and size of tune
        data_size = len(self.transformed_data)
        tune_size = int(data_size / 10)

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Sample for tune
        tune_splitter = []

        # Randomize a number between 0 and 10 and multiply by the index to randomly pick observations over data set
        for index in range(tune_size):
            tune_splitter.append(np.random.choice(a=10) + (10 * index))
        self.tune_data = self.transformed_data.iloc[tune_splitter]

        # Determine the remaining index that weren't picked for tune
        remainder = list(set(self.transformed_data.index) - set(tune_splitter))
        remainder_df = pd.DataFrame(self.transformed_data.iloc[remainder]['Class'])

        # Assign a random number
        remainder_df['Random_Number'] = np.random.randint(0, len(remainder), remainder_df.shape[0])

        # Sort over class and the random number
        remainder_df.sort_values(by=['Class', 'Random_Number'], inplace=True)
        remainder_df.reset_index(inplace=True)

        # Sample for CV
        for index in range(5):
            # Mod the index by 5 and there will be 5 remainder groups for the CV split
            splitter = remainder_df.loc[remainder_df.index % 5 == index]['index']

            # Update our attribute with the dictionary for this index
            self.test_split.update({
                index: self.transformed_data.iloc[splitter]
            })

    def cv_split_regression(self):
        """
        Function to split our transformed data into 10% tune and 5 cross tune splits for regression

        First this function splits out a tune set. The remainder is sampled 5 times to produce 5 cv splits.

        :return self.test_split: dict (of DataFrames), dictionary with keys (tune, 0, 1, 2, 3, 4) referring to the
            split transformed data
        """
        # Define base data size and size of tune and splits
        data_size = len(self.transformed_data)
        tune_size = int(data_size / 10)
        cv_size = int((data_size - tune_size) / 5)

        # The extra data will go to the first splits. The remainder of the length divided by 5 defines how many extra
        extra_data = int((data_size - tune_size) % cv_size)

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Sample for tune
        tune_splitter = np.random.choice(a=data_size, size=tune_size, replace=False)
        self.tune_data = self.transformed_data.iloc[tune_splitter]

        # Determine the remaining index that weren't picked for tune
        remainder = list(set(self.transformed_data.index) - set(tune_splitter))

        # CV Split
        for index in range(5):
            # For whatever number of extra data, we'll add one to each of those index
            if (index + 1) <= extra_data:
                # Sample for the CV size if extra data
                splitter = np.random.choice(a=remainder, size=(cv_size + 1), replace=False)

            else:
                # Sample for the CV size
                splitter = np.random.choice(a=remainder, size=cv_size, replace=False)

            # Define the remaining unsampled data points
            remainder = list(set(remainder) - set(splitter))

            # Update our attribute with the dictionary for this index
            self.test_split.update({
                index: self.transformed_data.iloc[splitter]
            })

    def cv_combine(self):
        """
        Function to combine the CV splits

        For each of the 5 CV splits, this function combines the other 4 splits and assigns it the same index as the
            left out split. This combined split is labeled as the train data set
        """
        # Loop through index
        for index in range(5):
            # Remove the current index from the train_index
            train_index = [train_index for train_index in [0, 1, 2, 3, 4] if train_index != index]
            train_data = pd.DataFrame()

            # For index in our train_index, append to the Data Frame
            for data_split_index in train_index:
                train_data = train_data.append(self.test_split[data_split_index])

            # Update train data with the combined CV
            self.train_split.update({index: train_data})
