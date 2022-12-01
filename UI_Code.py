################################################
# Sexual Violence In Media Prediction Software #
# Author: Angelo Kelvakis                      #
# DSC 478 FINAL                                #
# Current Version 1.0   11/16/22               #
################################################

# Load libraries
from os.path import exists
import pandas as pd
pd.options.mode.chained_assignment = None
import cmd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing


# This class houses all functions appropriate to execute the model.
class User_Interface:
    def load_data(self):
        '''
        :return: Pandas dataframe of 'full_media_df' file. Or returns None if filepath error
        '''
        # check that the data file is in the correct location
        if exists('full_media_df'):
            return pd.read_csv('full_media_df')
        else:
            # prompt the user to enter the correct filepath
            filenotfound = 'Y'
            while filenotfound == 'Y':
                print('\nERROR: DATA FILE NOT FOUND IN SAME LOCATION AS THIS SCRIPT. PLEASE COPY AND PASTE CORRECT FILE PATH')
                print('TO THE full_media_df FILE.')
                filepath = input('File Path: ')
                if exists(filepath):
                    df = pd.read_csv(filepath)
                    filenotfound = 'N'
                    print('File found, thank you!\n')
                else:
                    # If still not found, allow user to end the program
                    print('\nERROR: FILE NOT FOUND.')
                    print('Would you like to exit the program?')
                    A = input("Type Y or N: ").lower()
                    if A == 'Y':
                        filenotfound = 'N'
                        df = None
            return df

    def greetings(self):
        # print out Title of the program and program details
        rows = 6
        for i in range(0, rows):
            for j in range(0, i + 1):
                print("|", end=' ')
            print("-" * (rows+26))

        print("SEXUAL VIOLENCE IN MEDIA PREDICTION SOFTWARE")
        print("AUTHOR: Angelo Kelvakis         VERSION: 1.0")

        for i in range(rows + 1, 0, -1):
            for j in range(0, i - 1):
                print("|", end=' ')
            print("-" * (rows+26))
        # print out a short explanation of program
        print("\nThis program will take in media info and predicts whether or not that media contains sexual violence.")
        print('In order to properly predict your media, you must include:')
        print('Title, Release Year, Type of media, a brief synopsis (best if pulled from google), and Genres.')
        print('This model uses a stacked NB and ADABoost method of prediction and currently reports a 71% accuracy.')
        print('Follow prompts to enter information about your media of choice below. Thank you!\n')

    def pre_process(self, dataframe):
        # Subset data for searching the database first
        self.search = dataframe[['title','yearOfRelease','itemType','noRape', 'rapeMenDisImp',
       'sexHarOnScrn', 'sexAdultTeen', 'childSexAbuse', 'incest',
       'attemptedRape', 'rapeOffScrn', 'rapeOnScreen']]
        # convert title & itemType columns to lowercase for user input processing
        self.search.iloc[:, 0] = self.search.iloc[:, 0].str.lower()
        self.search.iloc[:, 2] = self.search.iloc[:, 2].str.lower()

        # collect genre names
        self.genres = ['action', 'adventure', 'animation', 'anime', 'anthology', 'art',
       'biography', 'childrens', 'comedy', 'contemporary', 'cooking', 'crime',
       'cult film', 'documentary', 'drama', 'dystopian', 'fantasy', 'guide',
       'historical', 'horror', 'lgbt', 'literary fiction', 'memoir', 'musical',
       'mystery', 'nonfiction', 'paranormal', 'poetry', 'reference', 'romance',
       'scifi', 'self-help', 'thriller', 'travel', 'western', 'war',
       'young adult']
        # convert to lowercase for user input text processing
        self.genres = [x.lower() for x in self.genres]
        # append genres with done, so logical matching passes through
        self.genres.append('done')

        # Pull out class labels for both models
        self.y_labs = dataframe['SV_binary']

        # Subset data for text frequency processing and NB model
        self.corpus = dataframe['overview']

        # Subset data for ADABoost model
        self.ADA_data = dataframe.iloc[:, 5:43]
        self.ADA_data = pd.concat([self.ADA_data, dataframe.iloc[:,-3:]], axis=1)


    def word_model(self):
        # Train test split data into 80-20%
        X_train, X_test, y_train, y_test = train_test_split(self.corpus,self.y_labs.values, test_size=0.20,random_state=0)
        # Count vectorize data to obtain frequency vector
        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(X_train)
        # Convert into tfidf data
        self.tfidf_transformer = TfidfTransformer()
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        # Run Naive Bayes model on train data
        self.tfidf_model = MultinomialNB()
        self.tfidf_model.fit(X_train_tfidf, y_train)
        # Use the NB model to predict on entire dataset
        X_full_counts = self.count_vect.transform(self.corpus)
        X_full_tfidf = self.tfidf_transformer.transform(X_full_counts)
        # create new column to bind to ADAboost model data
        self.predicted_overview = self.tfidf_model.predict(X_full_tfidf)


    def ADAboost_model(self):
        # Bind new predicted column to ada data
        self.ADA_data['predicted_from_overview'] = self.predicted_overview
        # min-max scale the data to account for the year column
        self.min_max_scaler = preprocessing.MinMaxScaler()
        df_norm = self.min_max_scaler.fit_transform(self.ADA_data.values)
        # convert scaled data back to pd df
        df_norm = pd.DataFrame(df_norm, columns=self.ADA_data.columns, index=self.ADA_data.index)
        # Re-run train test split with new randon_state as NB model
        X_train, X_test, y_train, y_test = train_test_split(df_norm, self.y_labs.values, test_size=0.20,random_state=2)
        # Load and run model on train data
        AB = AdaBoostClassifier()
        self.AB = AB.fit(X_train, y_train)

    def user_input_search(self):
        '''
        :return: program (while loop input that allows user_input_search() to run multiple times)
        outputs are = 'y','n','end','next'
        '''
        # prompt user for info that can be used to search for their media
        print("-" * 72)
        print('What is the name of movie, book, or TV show that you are trying to test?')
        u_title = input('Title: ').lower()
        print('What year was that media published?')
        self.u_year = int(input('Release year: '))
        print('What type of media is it? ')
        self.u_media = input('Movie, Book, or TV show: ').lower()
        # create user media dummy list
        if self.u_media.lower() == 'tv show':
            self.u_media_list = [1, 0, 0]
        elif self.u_media.lower() == 'book':
            self.u_media_list = [0, 1, 0]
        else:
            self.u_media_list = [0, 0, 1]
        # search DB for user inputs
        matching_row = self.search[((self.search['title'] == u_title) & (self.search['yearOfRelease'] == self.u_year) & (self.search['itemType'] == self.u_media))]
        # if there is no match, push user to prediction
        if len(matching_row) == 0:
            print("We did not find the media you were looking for, would you like to predict if this media contains sexual violence?")
            program = input("Type Y or N: ").lower()
            if program == 'y':
                program = 'next'
            else:
                program = 'end'
            return program
        # if there is a match, query user if they wish to search again
        else:
            SV_data = matching_row.iloc[:, 3:]
            SV_list_txt = ['No rape or sexual assault', 'Rape or sexual assault mentioned, discussed, implied',
                           'Sexual harassment (e.g non-consensual grabbing, touching, cat-calling)',
                           'Sexual relationship between adult and teenager', 'Child sex abuse', 'Incest', 'Attempted rape',
                           'Rape off-screen or strongly implied', 'Rape on-screen']
            # match the matching row data to the list of written out sexual violence items
            SV_found = [i for (i, v) in zip(SV_list_txt, SV_data.iloc[0]) if v]
            print(' ')
            print('-'*32)
            # print out matched data
            print('The media you selected contains:')
            for s in SV_found:
                print(s)
            # Query user if they wish to search again
            print('\nWould you like to search another media?')
            program = input("Type Y or N: ").lower()
            if program == "n":
                program = 'end'
            return program

    def user_input_overview(self):
        # If the users inputs were not found in the DB, query user for description
        print('-'*54)
        print('Copy and paste the synopsis of your media from google.')
        print('Or write a brief description. 2-3 sentences will do!')
        self.user_overview = input('Overview: ')

    def user_input_genres(self):
        # output the list of genres from the database in a column format
        cli = cmd.Cmd()
        print("List Of Genres:")
        print('-' * 80)
        cli.columnize(self.genres[:-1], displaywidth=80)
        print('-' * 80)
        print('\nLook through the list of potential genres for your media and enter any that fit!')
        print('When done, enter DONE.')
        genre_list = []
        g = 'n'
        while g != 'done':
            g = input('Genre: ').lower()
            correct_input = 0
            while correct_input == 0:
                # prevent user from misspelling the genre name
                if g not in self.genres:
                    print('-ERROR- GENRE NOT FOUND IN LIST, BE SURE TO MATCH SPELLING EXACTLY.')
                    g = input('Genre: ').lower()
                # Prevent user from entering same genre twice
                if g in genre_list:
                    print('-ERROR-: YOU HAVE ALREADY ENTERED THAT GENRE.')
                    g = input('Genre: ').lower()
                else:
                    correct_input = 1
            genre_list.append(g)
        # Make sure to remove the appended 'done' so it doesnt get matched
        genre_list = genre_list[:-1]
        genre_l = []
        # Collect the genres from the user submitted list
        for i in self.genres[:-1]:
            if i in genre_list:
                genre_l.append(1)
            else:
                genre_l.append(0)
        self.user_genre = genre_l

    def user_predict(self):
        # predict from user overview
        test1 = [self.user_overview]
        # transform user inputted text data
        user_overview_counts = self.count_vect.transform(test1)
        user_overview_tfidf = self.tfidf_transformer.transform(user_overview_counts)
        # output predicted class based on user overview data
        SV_text = self.tfidf_model.predict(user_overview_tfidf)[0]
        # Assemble user data
        user_data = [self.u_year] + self.user_genre + self.u_media_list + [SV_text]
        user_data_norm = self.min_max_scaler.transform([user_data])
        user_data_norm = pd.DataFrame(user_data_norm, columns=self.ADA_data.columns)
        # run the final prediction using the ADAboost model
        self.user_prediction = self.AB.predict(user_data_norm)[0]
        # Print out the model class as a written sentence to the user
        if self.user_prediction == 1:
            print('-'*57)
            print('MODEL OUTPUT:')
            print('Your {} may contain sexual violence.'.format(self.u_media))
            print('-' * 57)
        else:
            print('-' * 57)
            print('MODEL OUTPUT:')
            print('\nYour {} may NOT contain sexual violence.'.format(self.u_media))
            print('-' * 57)

# This is the fain function for running all the functions within the class. It will initialize databases, prompt the
# User for inputs, and run the predictive models, outputting a prediction to the user.
def Main():
    # Initiate the class
    UI = User_Interface()
    # print out the title
    UI.greetings()
    # load the data
    df = UI.load_data()
    # exit the program if user is having data loading issues
    if df is None:
        print('\n', '-' * 20, 'Thank you!', '-' * 20)
        return
    # slice up data into appropriate sub dfs
    UI.pre_process(df)
    # loop over prediction functions so user can test multiple medias
    input_predict = 'y'
    while input_predict == 'y':
        # Search the database for a pre-existing match to the user media
        program = 'y'
        while program == 'y':
            program = UI.user_input_search()
        if program == 'end':
            break
        # if not found, user enters the overview of their media
        UI.user_input_overview()
        # user enters genres
        UI.user_input_genres()
        # train the tfidf NB model
        UI.word_model()
        # train the ADAboost model
        UI.ADAboost_model()
        # Predict on all user inputs
        UI.user_predict()
        # Prompt user if they want to run again
        print('\nWould you like to test another media?')
        input_predict = input("Type Y or N: ").lower()
    # end of program
    print('\n', '-'*20, 'Thank you!', '-'*20)

# Cant get very far without this!
Main()


