-- ============================================================================
-- Migration 001: Initial Schema for WhatsApp RAG Assistant
-- Safe / Idempotent — uses CREATE TABLE IF NOT EXISTS
-- Run: mysql -u rag -p whatsapp_rag < migrations/001_initial_schema.sql
-- ============================================================================

SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- ── 1. users ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `users` (
    `id`             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `whatsapp_number` VARCHAR(20)    NOT NULL,
    `display_name`   VARCHAR(100)    DEFAULT NULL,
    `role`           ENUM('owner','admin','user','blocked') NOT NULL DEFAULT 'user',
    `is_active`      TINYINT(1)      NOT NULL DEFAULT 1,
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_whatsapp_number` (`whatsapp_number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 2. whatsapp_groups ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `whatsapp_groups` (
    `id`             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `group_jid`      VARCHAR(60)     NOT NULL,
    `group_name`     VARCHAR(200)    DEFAULT NULL,
    `is_active`      TINYINT(1)      NOT NULL DEFAULT 1,
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_group_jid` (`group_jid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 3. group_permissions ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `group_permissions` (
    `id`             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `group_id`       INT UNSIGNED    NOT NULL,
    `permission_key` VARCHAR(50)     NOT NULL COMMENT 'e.g. bot_enabled, admin_only, rag_enabled',
    `permission_val` VARCHAR(200)    NOT NULL DEFAULT 'true',
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_group_perm` (`group_id`, `permission_key`),
    CONSTRAINT `fk_gp_group` FOREIGN KEY (`group_id`) REFERENCES `whatsapp_groups` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 4. documents ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `documents` (
    `id`             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `filename`       VARCHAR(255)    NOT NULL,
    `file_path`      VARCHAR(500)    DEFAULT NULL,
    `file_type`      VARCHAR(20)     DEFAULT NULL COMMENT 'pdf, txt, md, etc.',
    `file_size`      INT UNSIGNED    DEFAULT NULL COMMENT 'bytes',
    `chunk_count`    INT UNSIGNED    DEFAULT 0,
    `status`         ENUM('pending','processing','done','error') NOT NULL DEFAULT 'pending',
    `uploaded_by`    INT UNSIGNED    DEFAULT NULL,
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_status` (`status`),
    CONSTRAINT `fk_doc_user` FOREIGN KEY (`uploaded_by`) REFERENCES `users` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 5. document_chunks ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `document_chunks` (
    `id`             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `document_id`    INT UNSIGNED    NOT NULL,
    `chunk_index`    INT UNSIGNED    NOT NULL DEFAULT 0,
    `chunk_text`     TEXT            NOT NULL,
    `qdrant_point_id` VARCHAR(64)   DEFAULT NULL COMMENT 'UUID of the vector point in Qdrant',
    `collection_name` VARCHAR(50)   DEFAULT 'personal_knowledge',
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_document` (`document_id`),
    KEY `idx_qdrant_point` (`qdrant_point_id`),
    CONSTRAINT `fk_chunk_doc` FOREIGN KEY (`document_id`) REFERENCES `documents` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 6. inventory_items ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `inventory_items` (
    `id`             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    `item_name`      VARCHAR(200)    NOT NULL,
    `category`       VARCHAR(100)    DEFAULT NULL,
    `description`    TEXT            DEFAULT NULL,
    `quantity`       INT             DEFAULT 1,
    `location`       VARCHAR(200)    DEFAULT NULL,
    `tags`           VARCHAR(500)    DEFAULT NULL COMMENT 'comma-separated tags',
    `added_by`       INT UNSIGNED    DEFAULT NULL,
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_category` (`category`),
    CONSTRAINT `fk_inv_user` FOREIGN KEY (`added_by`) REFERENCES `users` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 7. conversation_logs ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `conversation_logs` (
    `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `sender_number`  VARCHAR(20)     NOT NULL,
    `sender_name`    VARCHAR(100)    DEFAULT NULL,
    `chat_jid`       VARCHAR(60)     NOT NULL COMMENT 'private or group JID',
    `is_group`       TINYINT(1)      NOT NULL DEFAULT 0,
    `message_text`   TEXT            DEFAULT NULL,
    `bot_reply`      TEXT            DEFAULT NULL,
    `message_type`   VARCHAR(20)     DEFAULT 'text' COMMENT 'text, image, document, etc.',
    `rag_sources`    TEXT            DEFAULT NULL COMMENT 'JSON array of source refs used',
    `response_time_ms` INT UNSIGNED DEFAULT NULL,
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_sender` (`sender_number`),
    KEY `idx_chat` (`chat_jid`),
    KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 8. audit_logs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `audit_logs` (
    `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `actor_number`   VARCHAR(20)     NOT NULL COMMENT 'who performed the action',
    `action`         VARCHAR(50)     NOT NULL COMMENT 'e.g. adduser, setrole, blockuser',
    `target`         VARCHAR(200)    DEFAULT NULL COMMENT 'target user/group/entity',
    `details`        TEXT            DEFAULT NULL COMMENT 'JSON details of the action',
    `status`         ENUM('success','failed') NOT NULL DEFAULT 'success',
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_actor` (`actor_number`),
    KEY `idx_action` (`action`),
    KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── 9. failed_questions ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `failed_questions` (
    `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `sender_number`  VARCHAR(20)     NOT NULL,
    `question_text`  TEXT            NOT NULL,
    `chat_jid`       VARCHAR(60)     DEFAULT NULL,
    `attempted_sources` TEXT         DEFAULT NULL COMMENT 'JSON: what was searched',
    `best_score`     FLOAT           DEFAULT NULL COMMENT 'highest Qdrant score found',
    `is_resolved`    TINYINT(1)      NOT NULL DEFAULT 0,
    `resolution_note` TEXT           DEFAULT NULL,
    `created_at`     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `resolved_at`    TIMESTAMP       NULL DEFAULT NULL,
    PRIMARY KEY (`id`),
    KEY `idx_resolved` (`is_resolved`),
    KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ── Seed: Owner user ────────────────────────────────────────────────────────
INSERT INTO `users` (`whatsapp_number`, `display_name`, `role`, `is_active`)
VALUES ('6287877904270', 'Owner', 'owner', 1)
ON DUPLICATE KEY UPDATE `display_name` = VALUES(`display_name`);
