-- Glimpse3D job persistence table for the Supabase backend.
-- Mirrors the SQLAlchemy Job model (backend/app/models/job.py).
-- Apply once against your Supabase project (SQL editor or migration).

create table if not exists public.jobs (
    id          text primary key,                 -- job_id
    status      text not null default 'starting', -- starting | processing | completed | failed
    progress    double precision not null default 0,
    message     text,
    result      jsonb,
    error       text,
    warnings    jsonb not null default '[]'::jsonb,
    image_path  text,
    output_dir  text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Helpful indexes for the repository access patterns.
create index if not exists jobs_status_idx     on public.jobs (status);
create index if not exists jobs_created_at_idx on public.jobs (created_at desc);

-- NOTE on keys:
--   * Server-side writes use the SERVICE-ROLE key (bypasses RLS).
--   * The anon key is client-read only; gate it behind Row Level Security.
-- Example read-only RLS for the anon role (uncomment to enable):
-- alter table public.jobs enable row level security;
-- create policy "jobs anon read" on public.jobs for select to anon using (true);
