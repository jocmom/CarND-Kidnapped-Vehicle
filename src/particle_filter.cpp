/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#include "helper_functions.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	this->num_particles = 100;
	double init_weight = 1.;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for(int i=0; i<num_particles; ++i) {
		Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight};
		this->particles.push_back(p);
		this->weights.push_back(init_weight);
	}
	cout << "init" << endl;
	this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	const double delta_pos = velocity * delta_t; 
	const double f = velocity / yaw_rate;
	const double delta_theta = yaw_rate * delta_t;
	const bool is_moving_straight = fabs(yaw_rate) < 0.001;  
	default_random_engine gen;
	normal_distribution<double> dist_x(0., std_pos[0]);
	normal_distribution<double> dist_y(0., std_pos[1]);
	normal_distribution<double> dist_theta(0., std_pos[2]);


    for(auto it = particles.begin(); it != particles.end(); ++it) {
		if(is_moving_straight) {
			it->x += delta_pos * cos(it->theta) + dist_x(gen);
			it->y += delta_pos * sin(it->theta) + dist_y(gen);
			it->theta += dist_theta(gen);
		} 
		else {
			const double phi = it->theta + delta_theta;
			it->x += f * (sin(phi) - sin(it->theta)) + dist_x(gen); 
			it->y += f * (cos(it->theta) - cos(phi)) + dist_y(gen);
			it->theta = phi + dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size(); ++i) {
		double closestPrediction = __DBL_MAX__;
		for(int j=0; j<predicted.size(); ++j) {
			double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if(distance < closestPrediction) {
				closestPrediction = distance;
				// observations[i].id = predicted[j].id;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	const double std_x_square = 0.5/(std_x*std_x);
	const double std_y_square = 0.5/(std_y*std_y);
	const double d = 2.*M_PI*std_x*std_y;

	// Transformation from vehicle to map coordinate system for each particle
	for(vector<Particle>::iterator itP = particles.begin(); itP != particles.end(); ++itP) {
		vector<LandmarkObs> valid_landmarks;
		vector<LandmarkObs> transformed_observations;
		// translate and rotate observations 
		for(vector<LandmarkObs>::iterator itO = observations.begin(); itO != observations.end(); itO++) {
			double x = itP->x + itO->x * cos(itP->theta) - itO->y * sin(itP->theta);
			double y = itP->y + itO->x * sin(itP->theta) + itO->y * cos(itP->theta);
			LandmarkObs transformed_observation = {itO->id, x, y};
			transformed_observations.push_back(transformed_observation);
		}
		// select only valid landmarks in sensor range
		for(vector<Map::single_landmark_s>::iterator itL = map_landmarks.landmark_list.begin(); 
			itL != map_landmarks.landmark_list.end();
			++ itL) 
		{
			if(dist(itP->x, itP->y, itL->x_f, itL->y_f) <= sensor_range) {
				LandmarkObs valid_landmark = {itL->id_i, itL->x_f, itL->y_f};
				valid_landmarks.push_back(valid_landmark);
			}
		}
		if(valid_landmarks.size() == 0) {
			continue;
		}
		// Association
		this->dataAssociation(valid_landmarks, transformed_observations);

		// calculate particle weights
		double weight = 1.;
		for(vector<LandmarkObs>::iterator itO = transformed_observations.begin(); itO != transformed_observations.end(); itO++) {
			double diff_x = itO->x - valid_landmarks[itO->id].x;
			double diff_y = itO->y - valid_landmarks[itO->id].y;
			double diff_x_square = diff_x * diff_x;
			double diff_y_square = diff_y * diff_y;
			weight *= exp(-(diff_x_square*std_x_square + diff_y_square*std_y_square )) / d;
		}
		itP->weight = weight;
		this->weights[itP - particles.begin()] = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> dist_w(this->weights.begin(), this->weights.end());
	vector<Particle> updated_particles;
	updated_particles.resize(num_particles);
    
	for(int i=0; i<num_particles; i++) {
		updated_particles[i] = this->particles[dist_w(gen)];
	}
	this->particles = updated_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
