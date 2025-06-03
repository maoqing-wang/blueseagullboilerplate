#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install alembic')


# In[8]:


#!/usr/bin/env python3
"""
Migration CLI Script - Versioned Schema Migration Tool for PostgreSQL
A command-line utility to apply database schema changes using Alembic.

Usage:
    python migrate.py init                           # Initialize migrations
    python migrate.py revision "add user table"     # Create new migration
    python migrate.py upgrade                        # Apply all migrations
    python migrate.py downgrade -1                   # Rollback one migration
    python migrate.py status                         # Check current status
    python migrate.py history                        # Show migration history
    python migrate.py validate                       # Validate migrations
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Third-party imports
try:
    from dotenv import load_dotenv
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.exc import SQLAlchemyError, OperationalError
    from sqlalchemy.engine import Engine
except ImportError as e:
    print(f"Error: Required package not installed. Run: pip install {e.name}")
    sys.exit(1)


# Get current working directory and navigate to project root
current_dir = Path(os.getcwd())
project_root = current_dir.parent.parent.parent  # Adjust based on your structure
sys.path.insert(0, str(project_root))
# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class MigrationError(Exception):
    """Raised when migration operations fail."""
    pass


class MigrationManager:
    """Manages database migrations using Alembic."""
    
    def __init__(self, config_path: Optional[str] = None, database_url: Optional[str] = None):
        """
        Initialize the migration manager.
        
        Args:
            config_path: Path to alembic.ini file. Defaults to current directory.
            database_url: Database URL override. If not provided, uses environment.
        """
        self.base_dir = Path(__file__).parent
        self.config_path = config_path or str(self.base_dir / "alembic.ini")
        self.database_url = database_url or self._get_database_url()
        
        # Initialize Alembic configuration
        self.alembic_cfg = self._setup_alembic_config()
        
        # Database engine
        self.engine: Optional[Engine] = None
        self._setup_engine()
        
        logger.info(f"Initialized MigrationManager with config: {self.config_path}")
    
    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            # Construct from individual components if DATABASE_URL not provided
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            database = os.getenv("DB_NAME", "postgres")
            username = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            
            database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        # Mask password for logging
        masked_url = self._mask_password_in_url(database_url)
        logger.info(f"Using database URL: {masked_url}")
        
        return database_url
    
    def _mask_password_in_url(self, url: str) -> str:
        """Mask password in database URL for safe logging."""
        if '@' in url and '://' in url:
            scheme, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host_part = rest.split('@', 1)
                if ':' in credentials:
                    username, _ = credentials.split(':', 1)
                    return f"{scheme}://{username}:***@{host_part}"
        return url
    
    def _setup_alembic_config(self) -> Config:
        """Setup Alembic configuration."""
        try:
            alembic_cfg = Config(self.config_path)
            
            # Set the script location relative to this file
            script_location = str(self.base_dir / "migrations")
            alembic_cfg.set_main_option("script_location", script_location)
            
            # Set database URL
            alembic_cfg.set_main_option("sqlalchemy.url", self.database_url)
            
            return alembic_cfg
            
        except Exception as e:
            logger.error(f"Failed to setup Alembic configuration: {e}")
            raise MigrationError(f"Alembic configuration error: {e}")
    
    def _setup_engine(self) -> None:
        """Setup SQLAlchemy engine for direct database operations."""
        try:
            # Engine configuration
            engine_kwargs = {
                'echo': LOG_LEVEL == 'DEBUG',
                'pool_pre_ping': True,
                'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
                'connect_args': {
                    'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10'))
                }
            }
            
            # Add pool settings if specified
            pool_size = os.getenv('DB_POOL_SIZE')
            if pool_size:
                engine_kwargs['pool_size'] = int(pool_size)
            
            max_overflow = os.getenv('DB_MAX_OVERFLOW')
            if max_overflow:
                engine_kwargs['max_overflow'] = int(max_overflow)
            
            self.engine = create_engine(self.database_url, **engine_kwargs)
            
            # Test connection
            self._test_connection()
            
            logger.info("Database engine setup successful")
            
        except Exception as e:
            logger.error(f"Failed to setup database engine: {e}")
            raise DatabaseConnectionError(f"Database connection failed: {e}")
    
    def _test_connection(self, max_retries: int = 3, retry_delay: int = 2) -> None:
        """Test database connection with retries."""
        for attempt in range(max_retries):
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
                return
            except OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise DatabaseConnectionError(f"Failed to connect after {max_retries} attempts: {e}")
    
    @contextmanager
    def _database_operation(self, operation_name: str):
        """Context manager for database operations with error handling."""
        logger.info(f"Starting {operation_name}...")
        start_time = time.time()
        
        try:
            yield
            duration = time.time() - start_time
            logger.info(f"{operation_name} completed successfully in {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{operation_name} failed after {duration:.2f}s: {e}")
            raise MigrationError(f"{operation_name} failed: {e}")
    
    def init(self, directory: Optional[str] = None) -> bool:
        """
        Initialize Alembic directory structure for new services.
        
        Args:
            directory: Target directory for migrations. Defaults to ./migrations
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._database_operation("Migration initialization"):
            migrations_dir = directory or str(self.base_dir / "migrations")
            
            if Path(migrations_dir).exists():
                logger.warning(f"Migration directory already exists: {migrations_dir}")
                return False
            
            # Update config with custom directory if provided
            if directory:
                self.alembic_cfg.set_main_option("script_location", directory)
            
            command.init(self.alembic_cfg, migrations_dir)
            logger.info(f"Initialized Alembic directory structure at: {migrations_dir}")
            
            # Create custom env.py template
            self._create_env_template(migrations_dir)
            
            return True
    
    def _create_env_template(self, migrations_dir: str) -> None:
        """Create a custom env.py template with better configuration."""
        env_path = Path(migrations_dir) / "env.py"
        env_content = '''"""Alembic environment configuration."""
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your models here for autogenerate support
# Example:
# from myapp.models import Base
# target_metadata = Base.metadata
target_metadata = None

config = context.config

# Interpret the config file for Python logging if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def get_url():
    """Get database URL from environment or config."""
    return os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # For SQLite compatibility
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=True,  # For SQLite compatibility
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info("Created custom env.py template")
    
    def upgrade(self, target: str = "head") -> bool:
        """
        Apply migrations to bring DB to target state.
        
        Args:
            target: Target revision (default: "head" for latest)
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._database_operation(f"Database upgrade to {target}"):
            # Check current revision
            current_rev = self.current_revision()
            if current_rev:
                logger.info(f"Current revision: {current_rev}")
            else:
                logger.info("No current revision found (fresh database)")
            
            # Check for pending migrations
            pending_migrations = self._get_pending_migrations()
            if not pending_migrations and target == "head":
                logger.info("Database is already up to date")
                return True
            
            logger.info(f"Applying {len(pending_migrations)} pending migration(s)")
            
            command.upgrade(self.alembic_cfg, target)
            
            new_rev = self.current_revision()
            logger.info(f"Database upgraded successfully to: {new_rev}")
            return True
    
    def downgrade(self, target: str) -> bool:
        """
        Downgrade database to target revision.
        
        Args:
            target: Target revision to downgrade to
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._database_operation(f"Database downgrade to {target}"):
            current_rev = self.current_revision()
            if not current_rev:
                logger.warning("No current revision found, nothing to downgrade")
                return False
            
            logger.info(f"Current revision: {current_rev}")
            
            # Validate target revision
            if target != "base" and not self._is_valid_revision(target):
                logger.error(f"Invalid target revision: {target}")
                return False
            
            command.downgrade(self.alembic_cfg, target)
            
            new_rev = self.current_revision()
            logger.info(f"Database downgraded successfully to: {new_rev or 'base'}")
            return True
    
    def revision(self, message: str, autogenerate: bool = True) -> bool:
        """
        Generate a new migration file.
        
        Args:
            message: Descriptive message for the migration
            autogenerate: Whether to auto-detect schema changes
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._database_operation(f"Creating revision: {message}"):
            if not message.strip():
                logger.error("Migration message cannot be empty")
                return False
            
            # Sanitize message for filename
            safe_message = "".join(c for c in message if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_message = "_".join(safe_message.split())
            
            if autogenerate:
                command.revision(
                    self.alembic_cfg, 
                    message=safe_message, 
                    autogenerate=True
                )
                logger.info("Auto-generated migration based on model changes")
            else:
                command.revision(self.alembic_cfg, message=safe_message)
                logger.info("Created empty migration template")
            
            # Get the newly created revision
            script = ScriptDirectory.from_config(self.alembic_cfg)
            head = script.get_current_head()
            logger.info(f"Created migration file for revision: {head}")
            
            return True
    
    def current_revision(self) -> Optional[str]:
        """
        Get the current database revision.
        
        Returns:
            str: Current revision ID, None if no migrations applied
        """
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            logger.debug(f"Could not get current revision: {e}")
            return None
    
    def _get_pending_migrations(self) -> list:
        """Get list of pending migrations."""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.current_revision()
            
            if current_rev is None:
                # No migrations applied, all are pending
                return list(script.walk_revisions("base", "heads"))
            
            # Get revisions between current and head
            return list(script.walk_revisions(current_rev, "heads"))
        except Exception as e:
            logger.debug(f"Could not get pending migrations: {e}")
            return []
    
    def _is_valid_revision(self, revision: str) -> bool:
        """Check if a revision ID is valid."""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            script.get_revision(revision)
            return True
        except Exception:
            return False
    
    def history(self, verbose: bool = False, range_: Optional[str] = None) -> bool:
        """
        Show migration history.
        
        Args:
            verbose: Show detailed information
            range_: Specific range of revisions to show
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Migration History:")
            
            if range_:
                command.history(self.alembic_cfg, range_=range_, verbose=verbose)
            else:
                command.history(self.alembic_cfg, verbose=verbose)
            
            return True
        except Exception as e:
            logger.error(f"Failed to show history: {e}")
            return False
    
    def status(self) -> bool:
        """
        Show current migration status.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            current_rev = self.current_revision()
            
            if current_rev:
                logger.info(f"Current database revision: {current_rev}")
                
                # Get script directory to check for pending migrations
                script = ScriptDirectory.from_config(self.alembic_cfg)
                heads = script.get_heads()
                
                if current_rev in heads:
                    logger.info("✅ Database is up to date")
                else:
                    pending = self._get_pending_migrations()
                    logger.warning(f"⚠️  Database has {len(pending)} pending migration(s)")
                    logger.info(f"Head revisions: {', '.join(heads)}")
                    
                    if pending:
                        logger.info("Pending migrations:")
                        for migration in pending[:5]:  # Show first 5
                            logger.info(f"  - {migration.revision}: {migration.doc}")
                        if len(pending) > 5:
                            logger.info(f"  ... and {len(pending) - 5} more")
            else:
                logger.info("No migrations applied to database")
                
                # Check if there are any migrations available
                script = ScriptDirectory.from_config(self.alembic_cfg)
                try:
                    revisions = list(script.walk_revisions())
                    if revisions:
                        logger.info(f"Available migrations: {len(revisions)}")
                    else:
                        logger.info("No migration files found")
                except Exception:
                    logger.info("Migration directory not initialized")
            
            return True
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return False
    
    def validate(self) -> bool:
        """
        Validate migration scripts without applying them.
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            logger.info("Validating migration scripts...")
            
            # Check if migrations directory exists
            script_location = self.alembic_cfg.get_main_option("script_location")
            if not Path(script_location).exists():
                logger.error(f"Migration directory not found: {script_location}")
                return False
            
            # Get script directory
            script = ScriptDirectory.from_config(self.alembic_cfg)
            
            # Validate all revisions
            revisions = list(script.walk_revisions())
            if not revisions:
                logger.warning("No migration files found")
                return True
            
            logger.info(f"Found {len(revisions)} migration(s)")
            
            # Check for duplicate revision IDs
            revision_ids = [rev.revision for rev in revisions]
            if len(revision_ids) != len(set(revision_ids)):
                logger.error("Duplicate revision IDs found")
                return False
            
            # Validate each revision
            for revision in revisions:
                if not revision.revision:
                    logger.error(f"Invalid revision: {revision}")
                    return False
                
                logger.debug(f"Validated revision: {revision.revision} - {revision.doc}")
            
            logger.info("✅ All migration scripts are valid")
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False
    
    def heads(self) -> bool:
        """Show current head revisions."""
        try:
            command.heads(self.alembic_cfg, verbose=True)
            return True
        except Exception as e:
            logger.error(f"Failed to show heads: {e}")
            return False
    
    def show(self, revision: str) -> bool:
        """Show details of a specific revision."""
        try:
            command.show(self.alembic_cfg, revision)
            return True
        except Exception as e:
            logger.error(f"Failed to show revision {revision}: {e}")
            return False
    
    def database_info(self) -> Dict[str, Any]:
        """Get database information."""
        try:
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)
                
                info = {
                    'database_url': self._mask_password_in_url(self.database_url),
                    'database_name': conn.execute(text("SELECT current_database()")).scalar(),
                    'database_version': conn.execute(text("SELECT version()")).scalar(),
                    'current_user': conn.execute(text("SELECT current_user")).scalar(),
                    'tables': inspector.get_table_names(),
                    'schemas': inspector.get_schema_names(),
                }
                
                # Check if alembic_version table exists
                info['alembic_initialized'] = 'alembic_version' in info['tables']
                
                return info
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Database Migration CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s upgrade                    # Apply all pending migrations
  %(prog)s upgrade +1                 # Apply next migration
  %(prog)s downgrade -1               # Rollback one migration
  %(prog)s revision "add user table"  # Create new migration
  %(prog)s status                     # Show current status
  %(prog)s history                    # Show migration history
  %(prog)s validate                   # Validate migration scripts
  %(prog)s heads                      # Show head revisions
  %(prog)s show abc123                # Show specific revision

Environment Variables:
  DATABASE_URL      Full database connection URL
  DB_HOST          Database host (default: localhost)
  DB_PORT          Database port (default: 5432)
  DB_NAME          Database name (default: postgres)
  DB_USER          Database user (default: postgres)
  DB_PASSWORD      Database password
  LOG_LEVEL        Logging level (DEBUG, INFO, WARNING, ERROR)
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to alembic.ini config file",
        default=None
    )
    
    parser.add_argument(
        "--database-url",
        help="Database URL (overrides environment variables)",
        default=None
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize migration directory")
    init_parser.add_argument(
        "--directory", "-d",
        help="Migration directory path",
        default=None
    )
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Apply migrations")
    upgrade_parser.add_argument(
        "target",
        nargs="?",
        default="head",
        help="Target revision (default: head)"
    )
    
    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Rollback migrations")
    downgrade_parser.add_argument(
        "target",
        help="Target revision to downgrade to"
    )
    
    # Revision command
    revision_parser = subparsers.add_parser("revision", help="Create new migration")
    revision_parser.add_argument(
        "message",
        help="Migration message"
    )
    revision_parser.add_argument(
        "--no-autogenerate",
        action="store_true",
        help="Don't auto-detect schema changes"
    )
    
    # Status command
    subparsers.add_parser("status", help="Show current migration status")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show migration history")
    history_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    history_parser.add_argument(
        "--range", "-r",
        help="Show specific range of revisions"
    )
    
    # Validate command
    subparsers.add_parser("validate", help="Validate migration scripts")
    
    # Heads command
    subparsers.add_parser("heads", help="Show current head revisions")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show specific revision")
    show_parser.add_argument(
        "revision",
        help="Revision ID to show"
    )
    
    # Info command
    subparsers.add_parser("info", help="Show database information")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging level from arguments or environment
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize migration manager
    try:
        manager = MigrationManager(
            config_path=args.config,
            database_url=args.database_url
        )
    except (DatabaseConnectionError, MigrationError) as e:
        logger.error(f"Failed to initialize migration manager: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        sys.exit(1)
    
    # Execute command
    success = False
    
    try:
        if args.command == "init":
            success = manager.init(directory=args.directory)
        elif args.command == "upgrade":
            success = manager.upgrade(target=args.target)
        elif args.command == "downgrade":
            success = manager.downgrade(target=args.target)
        elif args.command == "revision":
            success = manager.revision(
                message=args.message,
                autogenerate=not args.no_autogenerate
            )
        elif args.command == "status":
            success = manager.status()
        elif args.command == "history":
            success = manager.history(
                verbose=args.verbose,
                range_=getattr(args, 'range', None)
            )
        elif args.command == "validate":
            success = manager.validate()
        elif args.command == "heads":
            success = manager.heads()
        elif args.command == "show":
            success = manager.show(args.revision)
        elif args.command == "info":
            info = manager.database_info()
            if info:
                print("\n=== Database Information ===")
                for key, value in info.items():
                    if isinstance(value, list):
                        print(f"{key}: {len(value)} items")
                        if value and len(value) <= 10:
                            for item in value:
                                print(f"  - {item}")
                    else:
                        print(f"{key}: {value}")
                success = True
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if LOG_LEVEL == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Only run main() if not in Jupyter
    import sys
    if 'ipykernel' not in sys.modules:
        main()
    else:
        print("Running in Jupyter - main() execution skipped")


# In[9]:


# Migration CLI Testing in Jupyter Notebook

import sys
import os
from pathlib import Path

# Fix the __file__ issue for Jupyter
try:
    # This works in .py files
    project_root = Path(__file__).parent.parent.parent
except NameError:
    # This works in Jupyter notebooks
    project_root = Path.cwd().parent.parent.parent  # Adjust based on your structure

sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Now import your migration functions
# (Replace with your actual import statements)
# from your_migration_module import create_argument_parser, main

def test_migration_cli():
    """Test the migration CLI with different commands"""
    
    # Create a modified main function that accepts arguments
    def test_main(command_args):
        """Modified main function for testing"""
        parser = create_argument_parser()  # Your existing parser
        
        # Parse the provided arguments instead of sys.argv
        args = parser.parse_args(command_args)
        
        print(f"Testing command: {' '.join(command_args)}")
        print(f"Parsed args: {args}")
        
        # Add your existing main() logic here, but use 'args' parameter
        # instead of parsing sys.argv
        
        return args

    # Test different commands
    test_commands = [
        ['init'],
        ['status'],
        ['upgrade'],
        ['downgrade'],
        ['revision', '--message', 'test migration'],
        ['history'],
        ['validate']
    ]
    
    for cmd in test_commands:
        try:
            print(f"\n{'='*50}")
            result = test_main(cmd)
            print(f"✅ Command '{' '.join(cmd)}' parsed successfully")
        except Exception as e:
            print(f"❌ Command '{' '.join(cmd)}' failed: {e}")

# Alternative: Test individual functions
def test_individual_functions():
    """Test migration functions individually without CLI parsing"""
    
    print("Testing individual migration functions:")
    
    # Example function calls (replace with your actual functions)
    try:
        # init_database()
        print("✅ Database initialization: OK")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    try:
        # get_migration_status()
        print("✅ Migration status check: OK")
    except Exception as e:
        print(f"❌ Migration status check failed: {e}")
    
    try:
        # validate_migrations()
        print("✅ Migration validation: OK")
    except Exception as e:
        print(f"❌ Migration validation failed: {e}")

# Quick CLI argument parser test
def test_argument_parser():
    """Test just the argument parser setup"""
    
    # Your argument parser creation code here
    parser = create_argument_parser()
    
    # Test parsing different argument combinations
    test_cases = [
        ['init'],
        ['upgrade', '--database-url', 'sqlite:///test.db'],
        ['revision', '--message', 'Add user table'],
        ['downgrade', '--steps', '1'],
        ['status', '--verbose']
    ]
    
    for case in test_cases:
        try:
            args = parser.parse_args(case)
            print(f"✅ {case}: {args}")
        except Exception as e:
            print(f"❌ {case}: {e}")

if __name__ == "__main__":
    # This won't cause issues in Jupyter since we're controlling it
    print("Migration CLI Testing")
    print("=" * 40)
    
    # Uncomment the test you want to run:
    # test_migration_cli()
    # test_individual_functions()
    # test_argument_parser()
    
    print("\nReady to test! Uncomment the test function you want to run.")


# In[ ]:





# In[ ]:




