import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import DataPreprocessing

class UserBasedCollaborativeFiltering:
    def __init__(self, user_data):
        """
        Initialize the User-Based Collaborative Filtering class.
        
        Parameters:
        - user_data: A Pandas DataFrame where rows represent users and columns represent features or items.
        """
        self.user_data = user_data
        self.similarity_matrix = None

        categorical_columns = ['Gender', 'Exercise', 'Weather Conditions']

        numerical_columns = ['Calories Burn', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity']

        preprocessor = DataPreprocessing(self.user_data, categorical_columns, numerical_columns)
        self.processed_data = preprocessor.preprocess_data()

    def compute_similarity(self):
        """
        Compute the cosine similarity between users in the dataset.
        
        Returns:
        - similarity_matrix: A matrix of cosine similarities between users.
        """
        # Compute the cosine similarity between users based on the processed features
        self.similarity_matrix = cosine_similarity(self.processed_data.drop(columns=['ID']))
        print("User similarity matrix computed.")
        return self.similarity_matrix

    def get_top_n_similar_users(self, user_id, n=5):
        """
        Get the top-N most similar users to the given user based on cosine similarity.
        
        Parameters:
        - user_id: The ID of the user for whom to find similar users.
        - n: The number of top similar users to return (default is 5).
        
        Returns:
        - A list of tuples representing the top-N similar users and their similarity scores.
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity() first.")

        # Get the index of the user
        user_idx = self.user_data.index[self.user_data['ID'] == user_id].tolist()[0]

        # Get the similarity scores for the given user
        user_similarity_scores = self.similarity_matrix[user_idx]
        print(f"Similarity scores for user {user_id}:")
        print(user_similarity_scores)
        
        # Get top-N similar users (excluding the current user)
        similar_users = np.argsort(user_similarity_scores)[::-1][1:]  # Sort in descending order and exclude itself
        similar_users = [(int(self.user_data.iloc[user]['ID']), float(user_similarity_scores[user])) for user in similar_users]
        
        print(f"Top {n} similar users for user {user_id}:")
        print(similar_users[:n])
        
        return similar_users[:n]

    def recommend_exercises(self, user_id, algorithm):
        """
        Recommend exercises for the given user based on similar users' exercises and include calorie information.
        
        Parameters:
        - user_id: The ID of the user for whom to recommend exercises.
        - algorithm: The selected recommendation algorithm to use.
        
        Returns:
        - recommended_exercises: A list of tuples with the format (exercise_name, calories_burned, sets, rest_time).
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity() first.")
        
        # Get the top 5 similar users
        similar_users = self.get_top_n_similar_users(user_id=user_id, n=5)
        
        # Recommend exercises based on similar users' choices
        recommended_exercises = []
        for similar_user_id, _ in similar_users:
            similar_user_data = self.user_data[self.user_data['ID'] == similar_user_id]
            similar_user_exercises = similar_user_data[['Exercise', 'Calories Burn', 'Exercise Intensity']].values.tolist()
            recommended_exercises.extend(similar_user_exercises)

        # Remove duplicates from the recommendation list
        unique_exercises = {}
        for exercise, calories, intensity in recommended_exercises:
            sets = self._calculate_sets(intensity)
            rest_time = self._calculate_rest_time(intensity, calories)
            unique_exercises[exercise] = (calories, sets, rest_time)

        # Ensure at least 5 exercises by adding frequent exercises if necessary
        if len(unique_exercises) < 5:
            all_exercises = self.user_data['Exercise'].value_counts().index.tolist()
            for ex in all_exercises:
                if ex not in unique_exercises:
                    # Calculate the average calories burned and intensity for this exercise
                    avg_calories = self.user_data[self.user_data['Exercise'] == ex]['Calories Burn'].mean()
                    avg_intensity = self.user_data[self.user_data['Exercise'] == ex]['Exercise Intensity'].mean()
                    sets = self._calculate_sets(avg_intensity if not np.isnan(avg_intensity) else 0)
                    rest_time = self._calculate_rest_time(avg_intensity if not np.isnan(avg_intensity) else 0, avg_calories if not np.isnan(avg_calories) else 0)
                    unique_exercises[ex] = (
                        avg_calories if not np.isnan(avg_calories) else 0,
                        sets,
                        rest_time
                    )

        # Convert back to a list of tuples
        recommended_exercises = [(ex, cal, sets, rest) for ex, (cal, sets, rest) in unique_exercises.items()]

        print(f"Recommended exercises for user {user_id}:")
        for exercise in recommended_exercises:
            print(exercise)
        
        return recommended_exercises[:5]

    def _calculate_sets(self, intensity):
        """
        Calculate the number of sets based on exercise intensity.
        
        Parameters:
        - intensity: The exercise intensity value (1–10 scale or similar).
        
        Returns:
        - Number of sets (1–5).
        """
        if intensity >= 8:
            return 5
        elif intensity >= 6:
            return 4
        elif intensity >= 4:
            return 3
        elif intensity >= 2:
            return 2
        else:
            return 1

    def _calculate_rest_time(self, intensity, calories):
        """
        Calculate the rest time between sets based on intensity and calories burned.
        
        Parameters:
        - intensity: The exercise intensity value (1–10 scale or similar).
        - calories: The number of calories burned during the exercise.
        
        Returns:
        - Rest time in seconds (minimum 20 seconds, maximum 300 seconds).
        """
        # Base rest time determined by intensity
        if intensity >= 8:
            base_rest = 110  # High intensity
        elif intensity >= 6:
            base_rest = 80
        elif intensity >= 4:
            base_rest = 50
        elif intensity >= 2:
            base_rest = 20
        else:
            base_rest = 10

        # Adjust rest time based on calories burned (higher calories -> longer rest)
        additional_rest = min(180, calories // 10)  # Add up to 3 minutes max

        return min(300, base_rest + additional_rest)

    def predict_fitness_score(self, user_id, feature_column, top_n=5):
            """
            Predict a user's fitness score (e.g., Calories Burned or Exercise Intensity) based on similar users.
            """
            top_similar_users = self.get_top_n_similar_users(user_id, n=top_n)

            # Get the original data for the similar users
            similar_user_ids = [user for user, _ in top_similar_users]
            similar_users_data = self.user_data[self.user_data['ID'].isin(similar_user_ids)]

            # Calculate the weighted average of the feature column based on similarity scores
            weights = np.array([score for _, score in top_similar_users])
            feature_values = similar_users_data[feature_column].values
            predicted_value = np.dot(weights, feature_values) / np.sum(weights)

            return predicted_value
        
    def get_user_profile_summary(self, user_id):
        """
        Return a summary of a user's profile (e.g., average Calories Burned, Heart Rate, BMI, etc.).
        Converts values to standard Python types (float, int) for better display.
        """
        user_profile = self.user_data[self.user_data['ID'] == user_id].squeeze()

        if user_profile.empty:
            return f"No user found with ID {user_id}"

        # Convert to native Python types (float, int) for better readability
        summary = {
            "Calories Burned": float(user_profile["Calories Burn"]),
            "Dream Weight": float(user_profile["Dream Weight"]),
            "Actual Weight": float(user_profile["Actual Weight"]),
            "Age": int(user_profile["Age"]),
            "Heart Rate": int(user_profile["Heart Rate"]),
            "BMI": float(user_profile["BMI"]),
            "Exercise Intensity": int(user_profile["Exercise Intensity"])
        }
        return summary
