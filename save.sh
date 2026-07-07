tar -czf /gemini-3/space/data/baiz/python_env_backup.tar.gz \
    $(python -c "import site; print(site.getsitepackages()[0])")   