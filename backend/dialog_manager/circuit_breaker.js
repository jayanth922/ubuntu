const axios = require('axios');

/**
 * Circuit Breaker implementation for service calls
 * Helps prevent cascading failures when a service is down
 */
class CircuitBreaker {
  /**
   * Create a new circuit breaker
   * @param {Object} options - Configuration options
   * @param {Number} options.failureThreshold - Number of failures before opening circuit (default: 3)
   * @param {Number} options.resetTimeout - Time in ms before trying again (default: 30000)
   * @param {Number} options.timeout - Request timeout in ms (default: 5000)
   */
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 3;
    this.resetTimeout = options.resetTimeout || 30000;
    this.timeout = options.timeout || 5000;
    
    // Circuit state
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.nextAttempt = Date.now();
    this.services = new Map();
  }
  
  /**
   * Register a service with the circuit breaker
   * @param {String} name - Service name
   * @param {String} baseUrl - Service base URL
   * @param {Object} options - Service-specific options (overrides defaults)
   * @returns {Object} - The service object
   */
  registerService(name, baseUrl, options = {}) {
    const service = {
      name,
      baseUrl,
      state: 'CLOSED',
      failureCount: 0,
      nextAttempt: Date.now(),
      failureThreshold: options.failureThreshold || this.failureThreshold,
      resetTimeout: options.resetTimeout || this.resetTimeout,
      timeout: options.timeout || this.timeout,
      fallback: options.fallback || null
    };
    
    this.services.set(name, service);
    return service;
  }
  
  /**
   * Set a fallback function for a service
   * @param {String} name - Service name
   * @param {Function} fallback - Fallback function that takes the same args as the original call
   */
  setFallback(name, fallback) {
    const service = this.services.get(name);
    if (service) {
      service.fallback = fallback;
    }
  }
  
  /**
   * Execute a request with circuit breaker protection
   * @param {String} serviceName - Name of the registered service
   * @param {String} endpoint - API endpoint (will be appended to baseUrl)
   * @param {Object} options - Axios request options
   * @param {Array} args - Arguments to pass to fallback function if circuit is open
   * @returns {Promise} - Promise resolving to response
   */
  async exec(serviceName, endpoint, options = {}, ...args) {
    const service = this.services.get(serviceName);
    
    if (!service) {
      throw new Error(`Service ${serviceName} not registered`);
    }
    
    // Check if circuit is open
    if (service.state === 'OPEN') {
      // Check if it's time to try again
      if (Date.now() > service.nextAttempt) {
        console.log(`Circuit for ${serviceName} is half-open, attempting request`);
        service.state = 'HALF-OPEN';
      } else {
        console.log(`Circuit for ${serviceName} is open, using fallback`);
        return this._handleOpenCircuit(service, ...args);
      }
    }
    
    try {
      // Set request timeout
      const requestOptions = {
        ...options,
        timeout: service.timeout
      };
      
      // Execute request
      const url = `${service.baseUrl}${endpoint}`;
      const response = await axios(url, requestOptions);
      
      // Reset on success if half-open
      if (service.state === 'HALF-OPEN') {
        this._closeCircuit(service);
      }
      
      return response;
      
    } catch (error) {
      return this._handleFailure(service, error, ...args);
    }
  }
  
  /**
   * Handle a request failure
   * @private
   */
  _handleFailure(service, error, ...args) {
    service.failureCount += 1;
    console.log(`Request to ${service.name} failed (${service.failureCount}/${service.failureThreshold}): ${error.message}`);
    
    // Check if we should open the circuit
    if (service.failureCount >= service.failureThreshold) {
      this._openCircuit(service);
    }
    
    // Use fallback if available
    return this._handleOpenCircuit(service, ...args);
  }
  
  /**
   * Open the circuit for a service
   * @private
   */
  _openCircuit(service) {
    service.state = 'OPEN';
    service.nextAttempt = Date.now() + service.resetTimeout;
    console.log(`Circuit for ${service.name} is now OPEN until ${new Date(service.nextAttempt).toISOString()}`);
  }
  
  /**
   * Close the circuit for a service
   * @private
   */
  _closeCircuit(service) {
    service.state = 'CLOSED';
    service.failureCount = 0;
    console.log(`Circuit for ${service.name} is now CLOSED`);
  }
  
  /**
   * Handle an open circuit by using fallback
   * @private
   */
  _handleOpenCircuit(service, ...args) {
    if (typeof service.fallback === 'function') {
      return service.fallback(...args);
    } else {
      throw new Error(`Service ${service.name} is unavailable and no fallback is defined`);
    }
  }
  
  /**
   * Get the status of all service circuits
   * @returns {Object} - Status of all services
   */
  getStatus() {
    const status = {};
    
    for (const [name, service] of this.services.entries()) {
      status[name] = {
        state: service.state,
        failureCount: service.failureCount,
        nextAttempt: service.state === 'OPEN' ? new Date(service.nextAttempt).toISOString() : null
      };
    }
    
    return status;
  }
  
  /**
   * Reset all circuits to closed state
   */
  resetAll() {
    for (const service of this.services.values()) {
      this._closeCircuit(service);
    }
  }
}

module.exports = CircuitBreaker;