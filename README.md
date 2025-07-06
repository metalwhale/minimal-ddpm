# minimal-ddpm
A minimal implementation of Denoising Diffusion Probabilistic Models for self-study

## Deployment
Change to [`deployment-local`](./deployment-local/) directory:
```console
host$ cd ./deployment-local/
```

Start and get inside the container:
```console
host$ docker compose up --build --remove-orphans -d
host$ docker compose exec minimal_ddpm bash
```

Run the program:
```console
minimal_ddpm# uv sync
minimal_ddpm# uv run python main.py
```
