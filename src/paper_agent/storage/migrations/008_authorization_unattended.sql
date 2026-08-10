ALTER TABLE authorization_grants
ADD COLUMN allow_unattended INTEGER NOT NULL DEFAULT 0
CHECK (allow_unattended IN (0, 1));
