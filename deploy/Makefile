# Manages Docker Compose services for production deployment

.PHONY: help up down restart logs build clean status shell-server

# Default target
help:
	@echo "Deployment Commands:"
	@echo "  up          - Start all services"
	@echo "  down        - Stop all services"
	@echo "  restart     - Restart all services"
	@echo "  logs        - Show logs for all services"
	@echo "  logs-f      - Follow logs for all services"
	@echo "  build       - Build ui and start services"
	@echo "  clean       - Stop services and remove volumes"
	@echo "  status      - Show service status"
	@echo "  shell-server - Open shell in server container"

# Start all services
up:
	bash run-deploy.sh run

run: up

# Stop all services
down:
	bash run-deploy.sh down

# Restart all services
restart-docker:
	bash run-deploy.sh restart

# Restart all services
restart: restart-docker logs-f

hard-restart: down up logs-f

# Show logs
logs:
	bash run-deploy.sh logs

# Follow logs
logs-f:
	bash run-deploy.sh logs-f

# Build ui and start services
build:
	@echo "Building ui..."
	cd ../ui && make build
	@echo "Starting services..."
	docker compose up -d

# Clean up - stop services and remove volumes
clean:
	docker compose down -v
	docker system prune -f

# Show service status
status:
	docker compose ps

# Open shell in server container
shell-server:
	docker compose exec gscodegen-server sh