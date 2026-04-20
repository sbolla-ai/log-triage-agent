# Nginx Troubleshooting Runbook

## Symptom: 502 Bad Gateway
1. Check if the upstream app is running: `systemctl status app`
2. Verify nginx can reach upstream: `curl localhost:8080/health`
3. Check nginx error log: `tail -f /var/log/nginx/error.log`

## Symptom: Service won't restart
1. Check for syntax errors: `nginx -t`
2. Look for port conflicts: `lsof -i :80`
3. Restart: `systemctl restart nginx`

## Symptom: SSL certificate errors
1. Check certificate expiry: `openssl x509 -in /etc/nginx/cert.pem -noout -dates`
2. Renew via certbot if expired: `certbot renew`
