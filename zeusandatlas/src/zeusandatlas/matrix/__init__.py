cat > src/zeusandatlas/matrix/__init__.py << 'EOF'
from .elementary import rowswap, rowscale, rowreplacement, rref

__all__ = ["rowswap", "rowscale", "rowreplacement", "rref"]
EOF
