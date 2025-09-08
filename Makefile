.PHONY: help up down restart logs build clean status

# Default target
help:
	cat Makefile

# Start all services
up:
	make build && cd deploy && make up && make logs-f

run: up

# Stop all services
down:
	cd deploy && make down

# Restart all services
restart:
	cd deploy && make restart

# Restart all services
hard-restart:
	make build && cd deploy && make down && make up && make logs-f

# Show logs
logs:
	cd deploy && make logs

# Follow logs
logs-f:
	cd deploy && make logs-f

# Clean up - stop services and remove volumes
clean-docker:
	cd deploy && docker compose down -v
	docker system prune -f
	@echo ""
	@echo "Done cleaning up"

# Show service status
status:
	docker compose ps

install:
	npm run install:all
	@echo ""
	@echo "Done installing"

build:
	npm run build
	@echo ""
	@echo "Done building"

start:
	npm run start

dev:
	npm run dev

clean:
	npm run clean
	@echo ""
	@echo "Done cleaning up"

list-scripts:
	npm run --workspace=server
	npm run --workspace=ui
	npm run --workspace=mcp-server

init-app-environment:
	bash ./scripts/init_app_environment.sh
	@echo ""
	@echo "Done initializing app environment"