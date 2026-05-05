"""
Admin command handler.

Processes /admin commands from WhatsApp.
Only admin/owner users can execute these commands.
All actions are audit-logged.

MVP Commands:
  /admin help
  /admin listusers
  /admin adduser <number> <name> <role>
  /admin setrole <number> <role>
  /admin blockuser <number>
  /admin unblockuser <number>
"""

from typing import Optional
from dataclasses import dataclass

from loguru import logger

from app.mysql_store import mysql_store


@dataclass
class CommandResult:
    """Result of an admin command."""
    success: bool
    message: str


VALID_ROLES = {"owner", "admin", "user", "blocked"}

HELP_TEXT = """📋 *Admin Commands*

/admin help — Show this help
/admin listusers — List all registered users
/admin adduser <number> <name> <role> — Add a user
/admin setrole <number> <role> — Change user role
/admin blockuser <number> — Block a user
/admin unblockuser <number> — Unblock a user

*Roles:* owner, admin, user, blocked

*Memory Commands:*
/remember <fact> — Save a memory
/memory search <query> — Search memories
/forget <keyword> — Delete a memory"""


class AdminCommands:
    """Admin command parser and executor."""

    def is_admin_command(self, text: str) -> bool:
        """Check if the text is an admin command."""
        return text.strip().lower().startswith("/admin")

    def execute(self, text: str, sender_number: str) -> CommandResult:
        """
        Parse and execute an admin command.

        Args:
            text: Full message text starting with /admin.
            sender_number: The sender's WhatsApp number.

        Returns:
            CommandResult with success status and response message.
        """
        # Check admin permission
        if not mysql_store.is_admin(sender_number):
            mysql_store.log_audit(
                actor_number=sender_number,
                action="admin_denied",
                details={"command": text},
                status="failed",
            )
            return CommandResult(
                success=False,
                message="⛔ You don't have admin permissions.",
            )

        # Parse command
        parts = text.strip().split()
        if len(parts) < 2:
            return CommandResult(success=False, message="❌ Invalid command. Use /admin help")

        subcommand = parts[1].lower()
        args = parts[2:]

        # Route to handler
        handlers = {
            "help": self._cmd_help,
            "listusers": self._cmd_list_users,
            "adduser": self._cmd_add_user,
            "setrole": self._cmd_set_role,
            "blockuser": self._cmd_block_user,
            "unblockuser": self._cmd_unblock_user,
        }

        handler = handlers.get(subcommand)
        if not handler:
            return CommandResult(
                success=False,
                message=f"❌ Unknown command: {subcommand}\nUse /admin help for available commands.",
            )

        result = handler(args, sender_number)

        # Audit log
        mysql_store.log_audit(
            actor_number=sender_number,
            action=subcommand,
            target=" ".join(args) if args else None,
            details={"command": text, "result": result.message},
            status="success" if result.success else "failed",
        )

        return result

    # ── Command Handlers ─────────────────────────────────────────────────

    def _cmd_help(self, args: list, sender: str) -> CommandResult:
        """Show help text."""
        return CommandResult(success=True, message=HELP_TEXT)

    def _cmd_list_users(self, args: list, sender: str) -> CommandResult:
        """List all registered users."""
        users = mysql_store.list_users()
        if not users:
            return CommandResult(success=True, message="📋 No users registered.")

        lines = ["📋 *Registered Users:*\n"]
        for u in users:
            status = "✅" if u.get("is_active") else "❌"
            role = u.get("role", "user")
            name = u.get("display_name", "—")
            number = u.get("whatsapp_number", "—")
            lines.append(f"{status} {name} ({number}) — {role}")

        return CommandResult(success=True, message="\n".join(lines))

    def _cmd_add_user(self, args: list, sender: str) -> CommandResult:
        """Add a new user: /admin adduser <number> <name> <role>"""
        if len(args) < 3:
            return CommandResult(
                success=False,
                message="❌ Usage: /admin adduser <number> <name> <role>\nRoles: owner, admin, user",
            )

        number = args[0]
        name = args[1]
        role = args[2].lower()

        if role not in VALID_ROLES or role == "blocked":
            return CommandResult(
                success=False,
                message=f"❌ Invalid role: {role}\nValid roles: owner, admin, user",
            )

        success = mysql_store.add_user(number, name, role)
        if success:
            return CommandResult(success=True, message=f"✅ User {name} ({number}) added as {role}")
        else:
            return CommandResult(success=True, message=f"ℹ️ User {number} already exists, name updated.")

    def _cmd_set_role(self, args: list, sender: str) -> CommandResult:
        """Change user role: /admin setrole <number> <role>"""
        if len(args) < 2:
            return CommandResult(
                success=False,
                message="❌ Usage: /admin setrole <number> <role>",
            )

        number = args[0]
        role = args[1].lower()

        if role not in VALID_ROLES:
            return CommandResult(
                success=False,
                message=f"❌ Invalid role: {role}\nValid: owner, admin, user, blocked",
            )

        # Check target exists
        user = mysql_store.get_user(number)
        if not user:
            return CommandResult(success=False, message=f"❌ User {number} not found.")

        success = mysql_store.set_role(number, role)
        if success:
            return CommandResult(success=True, message=f"✅ {user['display_name']} ({number}) role set to {role}")
        else:
            return CommandResult(success=False, message=f"❌ Failed to update role for {number}")

    def _cmd_block_user(self, args: list, sender: str) -> CommandResult:
        """Block a user: /admin blockuser <number>"""
        if len(args) < 1:
            return CommandResult(success=False, message="❌ Usage: /admin blockuser <number>")

        number = args[0]
        user = mysql_store.get_user(number)
        if not user:
            return CommandResult(success=False, message=f"❌ User {number} not found.")

        if user["role"] == "owner":
            return CommandResult(success=False, message="⛔ Cannot block the owner.")

        success = mysql_store.block_user(number)
        if success:
            return CommandResult(success=True, message=f"🚫 User {user['display_name']} ({number}) blocked.")
        else:
            return CommandResult(success=False, message=f"❌ Failed to block {number}")

    def _cmd_unblock_user(self, args: list, sender: str) -> CommandResult:
        """Unblock a user: /admin unblockuser <number>"""
        if len(args) < 1:
            return CommandResult(success=False, message="❌ Usage: /admin unblockuser <number>")

        number = args[0]
        user = mysql_store.get_user(number)
        if not user:
            return CommandResult(success=False, message=f"❌ User {number} not found.")

        success = mysql_store.unblock_user(number)
        if success:
            return CommandResult(success=True, message=f"✅ User {user['display_name']} ({number}) unblocked.")
        else:
            return CommandResult(success=False, message=f"❌ Failed to unblock {number}")


# Singleton instance
admin_commands = AdminCommands()
