import React, { useState, useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import '../styles/Admin.css';

// Register Chart.js components
Chart.register(...registerables);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const PROMETHEUS_URL = process.env.REACT_APP_PROMETHEUS_URL || 'http://localhost:9090';

function AdminDashboard() {
  const [metrics, setMetrics] = useState({});
  const [intents, setIntents] = useState([]);
  const [recentChats, setRecentChats] = useState([]);
  const [status, setStatus] = useState({});
  const [feedbackStats, setFeedbackStats] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [activeTab, setActiveTab] = useState('overview');
  
  const chatChartRef = useRef(null);
  const intentChartRef = useRef(null);
  const latencyChartRef = useRef(null);
  
  // Charts
  const chatChart = useRef(null);
  const intentChart = useRef(null);
  const latencyChart = useRef(null);
  
  // Fetch data on component mount and periodically
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch service status
        const statusRes = await fetch(`${API_URL}/health`);
        const statusData = await statusRes.json();
        setStatus(statusData);
        
        // Fetch metrics from Prometheus (in a real app)
        // For now, simulate with mock data
        const mockMetrics = {
          requests_per_minute: Math.floor(Math.random() * 50) + 10,
          avg_response_time: (Math.random() * 0.5 + 0.2).toFixed(2),
          success_rate: (Math.random() * 10 + 90).toFixed(1),
          active_sessions: Math.floor(Math.random() * 20) + 5
        };
        setMetrics(mockMetrics);
        
        // Mock intent distribution
        const mockIntents = [
          { intent: 'MakeUpdate', count: Math.floor(Math.random() * 100) + 50 },
          { intent: 'SetupPrinter', count: Math.floor(Math.random() * 80) + 20 },
          { intent: 'ShutdownComputer', count: Math.floor(Math.random() * 50) + 10 },
          { intent: 'SoftwareRecommendation', count: Math.floor(Math.random() * 70) + 30 },
          { intent: 'None', count: Math.floor(Math.random() * 30) + 10 }
        ];
        setIntents(mockIntents);
        
        // Mock recent chats
        const mockChats = [
          { id: '1', user: 'user1', timestamp: '2025-07-10 22:30:45', query: 'How do I update Ubuntu?', intent: 'MakeUpdate' },
          { id: '2', user: 'user2', timestamp: '2025-07-10 22:28:12', query: 'My printer is not working', intent: 'SetupPrinter' },
          { id: '3', user: 'user3', timestamp: '2025-07-10 22:25:33', query: 'How to install Chrome?', intent: 'MakeUpdate' },
          { id: '4', user: 'user4', timestamp: '2025-07-10 22:20:18', query: 'Best text editor for Ubuntu', intent: 'SoftwareRecommendation' },
          { id: '5', user: 'user5', timestamp: '2025-07-10 22:15:05', query: 'How to shutdown from terminal', intent: 'ShutdownComputer' }
        ];
        setRecentChats(mockChats);
        
        // Mock feedback stats
        const mockFeedback = {
          positive: Math.floor(Math.random() * 100) + 100,
          negative: Math.floor(Math.random() * 40) + 10,
          total: Math.floor(Math.random() * 140) + 110,
          rate: ((Math.random() * 20) + 80).toFixed(1)
        };
        setFeedbackStats(mockFeedback);
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching admin data:', error);
        setError('Failed to fetch dashboard data');
        setIsLoading(false);
      }
    };
    
    // Initial fetch
    fetchData();
    
    // Set up periodic refresh
    const interval = setInterval(fetchData, refreshInterval * 1000);
    
    // Clean up
    return () => clearInterval(interval);
  }, [refreshInterval]);
  
  // Initialize charts
  useEffect(() => {
    if (!isLoading && intents.length > 0) {
      // Destroy previous charts if they exist
      if (chatChart.current) chatChart.current.destroy();
      if (intentChart.current) intentChart.current.destroy();
      if (latencyChart.current) latencyChart.current.destroy();
      
      // Chat volume chart - last 24 hours
      const hours = Array.from(Array(24).keys()).map(h => `${h}:00`);
      const chatData = hours.map(() => Math.floor(Math.random() * 40) + 5);
      
      chatChart.current = new Chart(chatChartRef.current, {
        type: 'line',
        data: {
          labels: hours,
          datasets: [{
            label: 'Chat Volume',
            data: chatData,
            backgroundColor: 'rgba(233, 84, 32, 0.2)',
            borderColor: 'rgba(233, 84, 32, 1)',
            borderWidth: 1,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
      
      // Intent distribution chart
      intentChart.current = new Chart(intentChartRef.current, {
        type: 'doughnut',
        data: {
          labels: intents.map(i => i.intent),
          datasets: [{
            data: intents.map(i => i.count),
            backgroundColor: [
              'rgba(233, 84, 32, 0.7)',
              'rgba(119, 33, 111, 0.7)',
              'rgba(174, 167, 159, 0.7)',
              'rgba(47, 167, 212, 0.7)',
              'rgba(235, 218, 178, 0.7)'
            ],
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'bottom'
            }
          }
        }
      });
      
      // Response latency chart
      const latencyData = Array.from(Array(10).keys()).map(() => (Math.random() * 0.8 + 0.2).toFixed(2));
      
      latencyChart.current = new Chart(latencyChartRef.current, {
        type: 'bar',
        data: {
          labels: ['0.1s', '0.3s', '0.5s', '0.7s', '1s', '2s', '3s', '5s', '7s', '10s+'],
          datasets: [{
            label: 'Response Time Distribution',
            data: latencyData,
            backgroundColor: 'rgba(119, 33, 111, 0.7)',
            borderColor: 'rgba(119, 33, 111, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
    }
  }, [isLoading, intents]);
  
  const handleRefreshIntervalChange = (e) => {
    setRefreshInterval(Number(e.target.value));
  };
  
  if (error) {
    return (
      <div className="admin-error">
        <h1>Error</h1>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }
  
  return (
    <div className="admin-dashboard">
      <header className="admin-header">
        <h1>Ubuntu Support Chatbot Admin Dashboard</h1>
        <div className="admin-controls">
          <label>
            Refresh every:
            <select value={refreshInterval} onChange={handleRefreshIntervalChange}>
              <option value={10}>10 seconds</option>
              <option value={30}>30 seconds</option>
              <option value={60}>1 minute</option>
              <option value={300}>5 minutes</option>
            </select>
          </label>
          <span className="timestamp">Last updated: {new Date().toLocaleString()}</span>
        </div>
      </header>
      
      <nav className="admin-nav">
        <button 
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={activeTab === 'chats' ? 'active' : ''}
          onClick={() => setActiveTab('chats')}
        >
          Recent Chats
        </button>
        <button 
          className={activeTab === 'metrics' ? 'active' : ''}
          onClick={() => setActiveTab('metrics')}
        >
          Detailed Metrics
        </button>
        <button 
          className={activeTab === 'system' ? 'active' : ''}
          onClick={() => setActiveTab('system')}
        >
          System Status
        </button>
      </nav>
      
      <div className="admin-content">
        {isLoading ? (
          <div className="loading">Loading dashboard data...</div>
        ) : (
          <>
            {activeTab === 'overview' && (
              <div className="overview-tab">
                <div className="metrics-summary">
                  <div className="metric-card">
                    <h3>Requests/Min</h3>
                    <div className="metric-value">{metrics.requests_per_minute}</div>
                  </div>
                  <div className="metric-card">
                    <h3>Avg Response Time</h3>
                    <div className="metric-value">{metrics.avg_response_time}s</div>
                  </div>
                  <div className="metric-card">
                    <h3>Success Rate</h3>
                    <div className="metric-value">{metrics.success_rate}%</div>
                  </div>
                  <div className="metric-card">
                    <h3>Active Sessions</h3>
                    <div className="metric-value">{metrics.active_sessions}</div>
                  </div>
                </div>
                
                <div className="charts-container">
                  <div className="chart-box">
                    <h3>Chat Volume (24h)</h3>
                    <div className="chart-container">
                      <canvas ref={chatChartRef}></canvas>
                    </div>
                  </div>
                  
                  <div className="chart-box">
                    <h3>Intent Distribution</h3>
                    <div className="chart-container">
                      <canvas ref={intentChartRef}></canvas>
                    </div>
                  </div>
                </div>
                
                <div className="feedback-stats">
                  <h3>User Feedback</h3>
                  <div className="feedback-container">
                    <div className="feedback-metric">
                      <span>👍 Positive</span>
                      <span className="feedback-count">{feedbackStats.positive}</span>
                    </div>
                    <div className="feedback-metric">
                      <span>👎 Negative</span>
                      <span className="feedback-count">{feedbackStats.negative}</span>
                    </div>
                    <div className="feedback-metric">
                      <span>Total Ratings</span>
                      <span className="feedback-count">{feedbackStats.total}</span>
                    </div>
                    <div className="feedback-metric">
                      <span>Satisfaction Rate</span>
                      <span className="feedback-count">{feedbackStats.rate}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'chats' && (
              <div className="chats-tab">
                <h2>Recent Conversations</h2>
                <table className="chats-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>User</th>
                      <th>Timestamp</th>
                      <th>Query</th>
                      <th>Intent</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentChats.map(chat => (
                      <tr key={chat.id}>
                        <td>{chat.id}</td>
                        <td>{chat.user}</td>
                        <td>{chat.timestamp}</td>
                        <td>{chat.query}</td>
                        <td>
                          <span className={`intent-tag intent-${chat.intent.toLowerCase()}`}>
                            {chat.intent}
                          </span>
                        </td>
                        <td>
                          <button className="action-btn">View</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                
                <div className="pagination">
                  <button disabled>Previous</button>
                  <span>Page 1 of 1</span>
                  <button disabled>Next</button>
                </div>
              </div>
            )}
            
            {activeTab === 'metrics' && (
              <div className="metrics-tab">
                <h2>Detailed Performance Metrics</h2>
                
                <div className="metrics-controls">
                  <label>
                    Time Range:
                    <select defaultValue="24h">
                      <option value="1h">Last 1 hour</option>
                      <option value="6h">Last 6 hours</option>
                      <option value="24h">Last 24 hours</option>
                      <option value="7d">Last 7 days</option>
                    </select>
                  </label>
                  
                  <button className="metrics-refresh-btn">Refresh</button>
                </div>
                
                <div className="chart-box full-width">
                  <h3>Response Latency Distribution</h3>
                  <div className="chart-container">
                    <canvas ref={latencyChartRef}></canvas>
                  </div>
                </div>
                
                <div className="metrics-cards">
                  <div className="metrics-card">
                    <h3>Intent Classification</h3>
                    <div className="metrics-stats">
                      <div className="stat-item">
                        <span>Accuracy</span>
                        <span>87.5%</span>
                      </div>
                      <div className="stat-item">
                        <span>Avg Confidence</span>
                        <span>0.76</span>
                      </div>
                      <div className="stat-item">
                        <span>Low Confidence Rate</span>
                        <span>12.3%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="metrics-card">
                    <h3>RAG Retrieval</h3>
                    <div className="metrics-stats">
                      <div className="stat-item">
                        <span>Cache Hit Rate</span>
                        <span>32.1%</span>
                      </div>
                      <div className="stat-item">
                        <span>Avg Retrieval Time</span>
                        <span>231ms</span>
                      </div>
                      <div className="stat-item">
                        <span>Query Rewrite Rate</span>
                        <span>41.5%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'system' && (
              <div className="system-tab">
                <h2>System Status</h2>
                
                <div className="system-status">
                  <h3>Services</h3>
                  <div className="service-cards">
                    <div className={`service-card ${status.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
                      <h4>Dialog Manager</h4>
                      <div className="status-indicator"></div>
                      <span>{status.status || 'Unknown'}</span>
                    </div>
                    
                    <div className={`service-card ${status.services?.intent ? 'healthy' : 'unhealthy'}`}>
                      <h4>Intent Service</h4>
                      <div className="status-indicator"></div>
                      <span>{status.services?.intent ? 'Healthy' : 'Unhealthy'}</span>
                    </div>
                    
                    <div className={`service-card ${status.services?.rag ? 'healthy' : 'unhealthy'}`}>
                      <h4>RAG Service</h4>
                      <div className="status-indicator"></div>
                      <span>{status.services?.rag ? 'Healthy' : 'Unhealthy'}</span>
                    </div>
                    
                    <div className={`service-card ${status.services?.redis === 'connected' ? 'healthy' : 'unhealthy'}`}>
                      <h4>Redis</h4>
                      <div className="status-indicator"></div>
                      <span>{status.services?.redis === 'connected' ? 'Connected' : 'Disconnected'}</span>
                    </div>
                  </div>
                </div>
                
                <div className="admin-actions">
                  <h3>Administrative Actions</h3>
                  <div className="action-buttons">
                    <button className="admin-action-btn">Purge Cache</button>
                    <button className="admin-action-btn">Reset Metrics</button>
                    <button className="admin-action-btn dangerous">Clear All Conversations</button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default AdminDashboard;