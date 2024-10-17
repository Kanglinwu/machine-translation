# NGINX Load Balancing Configuration
## Overview
This NGINX configuration defines a load balancing setup for distributing incoming traffic across multiple backend servers (`server_1`, `server_2`, etc.) using the `least_conn` strategy, which directs traffic to the server with the fewest active connections.

## Configuration Details
1. **Upstream Block** (`backend`)
The `upstream` block defines the backend servers that NGINX will balance traffic across. The `least_conn` directive ensures that incoming requests are routed to the backend server with the fewest active connections at the time of the request.

```nginx
upstream backend {
  least_conn;
  server server_1:5050;
  server server_2:5050;
  server server_3:5050;
  server server_4:5050;
  server server_5:5050;
}
```
* `least_conn`: This method is used to distribute traffic by sending requests to the server with the least number of active connections, which helps balance the load effectively during high traffic periods.
* `server server_n:5050`: Defines the individual backend servers (`server_1`, `server_2`, etc.), each listening on port `5050`.
2. **Server Block**
The `server` block listens on port `80` (HTTP) and forwards all incoming requests to the backend defined in the `upstream` block.

```nginx
server {
  listen 80;
  include /etc/nginx/mime.types;
  location / {
    proxy_pass http://backend/;
  }
}
```
* `listen 80`: This instructs NGINX to listen for HTTP requests on port `80`.
* `include /etc/nginx/mime.types`: Specifies the MIME types for proper content type handling.
* `location /`: Defines the root location for handling requests. All requests to `/` will be proxied to the `backend` group of servers.
* `proxy_pass http://backend/`: Proxies the incoming requests to the `backend` group of servers specified in the `upstream`   block.

## Load Balancing Method
* **Least Connections** (`least_conn`): This method ensures that the backend server with the fewest active connections is selected to handle the request, which helps improve performance and manage traffic evenly across servers.

## Setup Instructions
1. **Save the Configuration**: Save the above configuration in your NGINX configuration file (e.g., `/etc/nginx/nginx.conf` or `/etc/nginx/conf.d/load_balancer.conf`).
2. **Ensure Backend Servers are Running**: Ensure that the backend servers (`server_1`, `server_2`, etc.) are properly configured and listening on port `5050`.
3. **Restart NGINX**  : After configuring the file, restart NGINX for the changes to take effect:
```bash
sudo systemctl restart nginx
```
4. **Verify Configuration**: Check if the configuration is valid using the following command:
```bash
sudo nginx -t
```

## Notes
* **Scaling**: You can scale this configuration by adding or removing backend servers within the `upstream` block.
* **Custom Port**: If the backend servers listen on a different port, update the `:5050` part accordingly.
* **Secure Traffic**: If handling sensitive data, it's recommended to configure SSL (HTTPS) for secure communication by listening on port `443` with appropriate SSL settings.

## Troubleshooting
* NGINX Not Restarting: If NGINX fails to restart, check for syntax errors using `nginx -t`.
* Connection Issues: Ensure the backend servers are reachable from the machine where NGINX is running.

## Additional Resources
* NGINX Documentation
* Load Balancing Methods in NGINX




