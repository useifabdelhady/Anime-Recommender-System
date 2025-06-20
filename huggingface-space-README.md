---
title: Anime Recommendation System
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Anime Recommendation System

This Space hosts a hybrid anime recommendation system that combines user-based and content-based filtering to provide personalized anime recommendations.

## How to Use

1. Enter a user ID in the input field
2. Click "Get Recommendations"
3. View your personalized anime recommendations

## About the Model

This recommendation system uses a hybrid approach:

- **User-based collaborative filtering**: Finds similar users based on rating patterns
- **Content-based filtering**: Recommends anime similar to those the user has highly rated
- **Hybrid combination**: Merges both approaches for better recommendations

## Technical Details

- Built with TensorFlow for the recommendation model
- Flask web application for the user interface
- Containerized with Docker for easy deployment

## Dataset

The model is trained on anime ratings and metadata, including:
- User ratings
- Anime details (title, genre, etc.)
- Anime synopses for content-based recommendations

## Limitations

- The system requires a valid user ID that exists in the training data
- Performance depends on the amount of rating data available for a user
- The model is based on a specific anime dataset and may not include very recent anime