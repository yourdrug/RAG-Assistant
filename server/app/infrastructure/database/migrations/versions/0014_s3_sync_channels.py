"""S3 prefixes as sync channels."""

from alembic import op


revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def replace_check(kinds: str) -> None:
    op.drop_constraint("sync_channels_storage_kind_check", "sync_channels", type_="check")
    op.create_check_constraint(
        "sync_channels_storage_kind_check", "sync_channels", f"storage_kind IN ({kinds})"
    )


def upgrade() -> None:
    replace_check("'github', 's3'")


def downgrade() -> None:
    bound = (
        op.get_bind().exec_driver_sql("SELECT count(*) FROM sync_channels WHERE storage_kind = 's3'").scalar()
    )

    if bound:
        msg = f"{bound} S3 sync channels exist; delete them before downgrading"
        raise RuntimeError(msg)

    replace_check("'github'")
