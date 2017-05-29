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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"
#include "map.h"

using namespace std;

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

    for(auto& p:particles) {
		if(is_moving_straight) {
			p.x += delta_pos * cos(p.theta) + dist_x(gen);
			p.y += delta_pos * sin(p.theta) + dist_y(gen);
			p.theta += dist_theta(gen);
		} 
		else {
			const double phi = p.theta + delta_theta;
			p.x += f * (sin(phi) - sin(p.theta)) + dist_x(gen); 
			p.y += f * (cos(p.theta) - cos(phi)) + dist_y(gen);
			p.theta = phi + dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(auto& o:observations) {
		double closestPrediction = __DBL_MAX__;
		int idx = 0;
		for(const auto& p:predicted) {
			double distance = dist(p.x, p.y, o.x, o.y);
			if(distance < closestPrediction) {
				closestPrediction = distance;
				o.id = idx;
			}
			idx++;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	const double std_x_square = 0.5/(std_x*std_x);
	const double std_y_square = 0.5/(std_y*std_y);
	const double d = 2.*M_PI*std_x*std_y;
	int idx = 0;

	// Transformation from vehicle to map coordinate system for each particle
	for(auto& p:particles) {
		vector<LandmarkObs> valid_landmarks;
		vector<LandmarkObs> transformed_observations;
		// translate and rotate observations 
		for(const auto& o:observations) {
			double x = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
			double y = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
			LandmarkObs transformed_observation = {o.id, x, y};
			transformed_observations.push_back(transformed_observation);
		}
		// select only valid landmarks in sensor range
		for(const auto& l:map_landmarks.landmark_list) {
			if(dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range) {
				LandmarkObs valid_landmark = {l.id_i, l.x_f, l.y_f};
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
		for(const auto& o:transformed_observations) {
			double diff_x = o.x - valid_landmarks[o.id].x;
			double diff_y = o.y - valid_landmarks[o.id].y;
			double diff_x_square = diff_x * diff_x;
			double diff_y_square = diff_y * diff_y;
			weight *= exp(-(diff_x_square*std_x_square + diff_y_square*std_y_square )) / d;
		}
		p.weight = weight;
		this->weights[idx++] = weight;
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

	for(auto& p:updated_particles) {
		p = this->particles[dist_w(gen)]; 
	}
	this->particles = updated_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
