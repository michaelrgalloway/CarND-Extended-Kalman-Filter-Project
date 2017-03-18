#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  //measurement matrix

  //Initialize the linear laser measurement extraction matix to pull out the 
  //position values(the 1's'), we dont pull out the velocity values since 
  //we infer those from the process operations
	H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //Initialize the state matrix, this holds our linear model (theoretical) in matrix form. 
  //When determining our estimated next state X, we multiple Xk-1 by F to obtain our predicted next state
  //Not here but if we also wanted to guage acceleration we can also introduce a B matrix that can 
  //further update X that incorporates change in velocity
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;

 //Initialize state covariance matrix with some values, these values will quickly change as we start running
 //the algorithm
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;


}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_)
  {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;


    //We since our cycles rely on previous states of X, and if we dont have a previous X
    //then we'll have to end here after just simply accepting our inputs as our X. 

    //We have one X state vector and it stores the values of position and velocity in 
    //cartesian coordinates, however we are accepting sensor values in both cartesian 
    //and polor coordinates. So depending on what type of data is coming in, we need either 
    //accept the data as-is, or convert to cartesion from polar. After we obtain cartesian
    //coordinates, we set X directly to those values.
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];

      ekf_.x_ << rho * cos(phi), rho * sin(phi), 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
   //Noise values for noise covariance Q, constant at 9. This was predetermined for this project.
  const float noise_ax = 9.0;
  const float noise_ay = 9.0;

  //obtain delta time since our last reading and as well as updating our prev time variable.
  //dt is not always a constant and we should be able to accept measurements at any time so we 
  //update the state transition matrix with the actual dt values so we obtain good calculations 
  //during our prediction step. 

  //We also update the Q noise covariance matrix with our real delta t's. The Q matrix keeps the
  //state covariance matrix from becoming too small or zero
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; 
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
      0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
      dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  //Now depending on if the sensor type is radar or laser, our methods will be full linear 
  //or will need some more operations to forge linear relationships out of non-linear ones. 
  

  //If the measurement type is radar or polar, we will need to use a Jacobian Matrix 
  //(a matrix populated with partial derivitives) to let us use polar values in our operations
  //
  //If the laser measurement is present, then we can use the usual extraction matrix H
  //
  //For both sensor types, the values for measurement covariance matrix R is provided above.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else
  {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
    // Laser updates
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
