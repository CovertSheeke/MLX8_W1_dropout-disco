# MLX8_W1_dropout-disco

A simple PostgreSQL stack with vector support, pgAdmin, and CloudBeaver for database management.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/)

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/CovertSheeke/MLX8_W1_dropout-disco.git
   cd MLX8_W1_dropout-disco
   cd postgresql
   ```

2. **Start the services:**
   ```sh
   docker compose up -d
   ```

3. **Access the services:**
   - **PostgreSQL:**  
     Host: `localhost`  
     Port: `5432`  
     User: `example`  
     Password: `example`
   - **pgAdmin:**  
     URL: [http://localhost:5050](http://localhost:5050)  
     Email: `admin@admin.com`  
     Password: `admin`
   - **CloudBeaver:**  
     URL: [http://localhost:8978](http://localhost:8978)

4. **Stop the services:**
   ```sh
   docker compose down
   ```

## Data Persistence

Data is stored in Docker volumes and will persist between restarts.

## Notes

- The PostgreSQL image includes [pgvector](https://github.com/pgvector/pgvector) extension for vector search.
- You can manage your databases using either pgAdmin or CloudBeaver.

---

This is the repository for MLX8 week 1.

In this repo we will be attempting to create a upvote prediction model.