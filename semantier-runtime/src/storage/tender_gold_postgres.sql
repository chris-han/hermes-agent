-- Explicit workspace-owned Tender annotation migration; no runtime DDL.
CREATE TABLE IF NOT EXISTS tender_gold_packages (
 id TEXT PRIMARY KEY, org TEXT NOT NULL, workspace TEXT NOT NULL, hash TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tender_gold_assignments (
 package TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL, payload TEXT NOT NULL,
 PRIMARY KEY(package,actor,role)
);
CREATE TABLE IF NOT EXISTS tender_gold_records (
 hash TEXT PRIMARY KEY, package TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL,
 kind TEXT NOT NULL, case_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
 UNIQUE(package,actor,role,kind,case_id,revision)
);
CREATE TABLE IF NOT EXISTS tender_gold_drafts (
 package TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL, case_id TEXT NOT NULL,
 revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(package,actor,role,case_id)
);
