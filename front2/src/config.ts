/**
 * Application configuration
 * Only uses PPO backend for scheduling
 */

// PPO Backend Configuration
const envPPOApiUrl = import.meta.env.VITE_PPO_API_URL;

// Default to localhost:8000 for PPO backend
export const PPO_API_URL = envPPOApiUrl || "http://localhost:8000";

// For backward compatibility - point all API calls to PPO backend
// Note: Components using these endpoints may need to be updated
// as PPO backend only supports /health and /schedule endpoints
export const API_BASE_URL = PPO_API_URL;

// Development note: PPO backend loads jobs directly from database
// Manual job/machine management endpoints are not available