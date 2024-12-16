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
        
        # Get top-N similar users (excluding the current user)
        similar_users = np.argsort(user_similarity_scores)[::-1][1:]  # Sort in descending order and exclude itself
        similar_users = [(int(self.user_data.iloc[user]['ID']), float(user_similarity_scores[user])) for user in similar_users]
        
        return similar_users[:n]

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

    def recommend_exercises(self, user_id):
        """
        Recommend exercises for the given user based on similar users' exercises and include calorie information.
        
        Parameters:
        - user_id: The ID of the user for whom to recommend exercises.
        
        Returns:
        - recommended_exercises: A list of tuples with the format (exercise_name, calories_burned, sets).
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
            unique_exercises[exercise] = (calories, sets)

        # Ensure at least 5 exercises by adding frequent exercises if necessary
        if len(unique_exercises) < 5:
            all_exercises = self.user_data['Exercise'].value_counts().index.tolist()
            for ex in all_exercises:
                if ex not in unique_exercises:
                    # Calculate the average calories burned and intensity for this exercise
                    avg_calories = self.user_data[self.user_data['Exercise'] == ex]['Calories Burn'].mean()
                    avg_intensity = self.user_data[self.user_data['Exercise'] == ex]['Exercise Intensity'].mean()
                    sets = self._calculate_sets(avg_intensity if not np.isnan(avg_intensity) else 0)
                    unique_exercises[ex] = (
                        avg_calories if not np.isnan(avg_calories) else 0,
                        sets
                    )

        # Convert back to a list of tuples
        recommended_exercises = [(ex, cal, sets) for ex, (cal, sets) in unique_exercises.items()]

        return recommended_exercises[:5]

    def get_exercise_recommendations(self, user_id, top_n_similar_users=5):
        """
        Recommend exercises to a user based on exercises performed by similar users.
        
        Parameters:
        - user_id: The ID of the user for whom to recommend exercises.
        - top_n_similar_users: Number of similar users to consider (default is 5).

        Returns:
        - A list of exercise recommendations with at least 5 items.
        """
        top_similar_users = self.get_top_n_similar_users(user_id, top_n_similar_users)
        similar_user_ids = [user for user, _ in top_similar_users]
        similar_users_data = self.user_data[self.user_data['ID'].isin(similar_user_ids)]

        # Get the most frequent exercises performed by similar users
        recommended_exercises = similar_users_data['Exercise'].value_counts().index.tolist()

        # Ensure at least 5 exercises by adding frequent exercises from all users if necessary
        if len(recommended_exercises) < 5:
            all_exercises = self.user_data['Exercise'].value_counts().index.tolist()
            additional_exercises = [ex for ex in all_exercises if ex not in recommended_exercises]
            recommended_exercises.extend(additional_exercises)

        return recommended_exercises[:5]
