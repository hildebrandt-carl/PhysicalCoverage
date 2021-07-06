class PID:
  def __init__(self, Kp_in=-1.0, Ki_in=-1.0, Kd_in=-1.0, rate_in=-1.0):

    # Variable to set the rate
    self.rate = rate_in

    # Calculate the time between intervals
    self.dt = 1.0/self.rate

    # Setting the PID parameters
    self.Kp = Kp_in
    self.Ki = Ki_in
    self.Kd = Kd_in

    # Variables used for the controller
    self.integral = 0.0
    self.previous_error = 0.0


  def set_constants(self, Kp_in, Ki_in, Kd_in):
    # Setting the PID constants
    self.Kp = Kp_in
    self.Ki = Ki_in
    self.Kd = Kd_in


  # Clear the integral and previous error
  def remove_buildup(self):
    self.integral = 0.0
    self.previous_error = 0.0


  # This is the main loop of this class
  def get_output(self, setpoint, current_output):

    # Generated output
    output = 0.0

    # Run the Controller
    error = setpoint - current_output
    self.integral = self.integral + error * self.dt
    derivative = (error - self.previous_error)/self.dt
    output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
    self.previous_error = error 

    # Return the output
    return output