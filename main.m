% 1. Loading movie ratings dataset 

fprintf('Loading movie ratings dataset.\n\n');
load ('movies_data.mat');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;







% 2. Collaborative Filtering Cost Function
load ('movie_parameters.mat');% X and Theta

%  Reducing the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = costfunction([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);
           
fprintf(['Cost at loaded parameters: %f '], J);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;






% 3. Entering ratings for a new user

movieList = loadMovieList();
my_ratings = zeros(1682, 1);
my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
pause;







% 4.Learning Movie Ratings

fprintf('\nTraining collaborative filtering...\n');

%  Loading data
load('movies_data.mat');

% Adding the rating of new user to movies_data
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalizing Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Setting Initial Parameters
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(costfunction(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);


fprintf('Learning of X and Theta completed.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;






% 5.Recommending Movies
p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end