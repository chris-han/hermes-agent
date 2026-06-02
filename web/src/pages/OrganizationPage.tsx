import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRightLeft,
  Check,
  Network,
  RefreshCw,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { api, type OrganizationSettingsResponse } from "@/lib/api";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

function statusTone(
  status: string,
): "secondary" | "warning" | "destructive" | "success" | "outline" {
  switch (status) {
    case "active":
      return "success";
    case "pending":
      return "warning";
    case "suspended":
    case "revoked":
      return "destructive";
    default:
      return "secondary";
  }
}

export default function OrganizationPage() {
  const [data, setData] = useState<OrganizationSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [inviteSubmitting, setInviteSubmitting] = useState(false);
  const [createOrgId, setCreateOrgId] = useState("");
  const [joinOrgId, setJoinOrgId] = useState("");
  const [orgName, setOrgName] = useState("");
  const [invitee, setInvitee] = useState("");
  const { toast, showToast } = useToast();
  const { t } = useI18n();

  const load = useCallback(() => {
    setLoading(true);
    api
      .getOrganizationSettings()
      .then(setData)
      .catch((error) => showToast(`${t.status.error}: ${error}`, "error"))
      .finally(() => setLoading(false));
  }, [showToast, t.status.error]);

  useEffect(() => {
    load();
  }, [load]);

  const current = data?.organization ?? null;
  const memberships = data?.memberships ?? [];
  const members = data?.members ?? [];
  const auditEvents = data?.audit_events ?? [];
  const currentOrgId = current?.organization_id ?? null;
  const activeMembers = useMemo(
    () => members.filter((item) => item.membership_status === "active"),
    [members],
  );
  const pendingMembers = useMemo(
    () => members.filter((item) => item.membership_status === "pending"),
    [members],
  );
  const statusLabel = useCallback(
    (status: string) =>
      t.organization.statuses[
        (["unassigned", "pending", "active", "suspended", "revoked"].includes(status)
          ? status
          : "unassigned") as keyof typeof t.organization.statuses
      ],
    [t.organization.statuses],
  );

  const runAction = useCallback(
    async (fn: () => Promise<OrganizationSettingsResponse>, successMessage: string) => {
      setSubmitting(true);
      try {
        const next = await fn();
        setData(next);
        showToast(successMessage, "success");
      } catch (error) {
        showToast(`${t.status.error}: ${error}`, "error");
      } finally {
        setSubmitting(false);
      }
    },
    [showToast, t.status.error],
  );

  const handleCreateOrg = async () => {
    if (!orgName.trim()) {
      showToast(t.organization.nameRequired, "error");
      return;
    }
    await runAction(
      () =>
        api.joinOrganization({
          organization_id: createOrgId.trim() || undefined,
          organization_name: orgName.trim(),
          create: true,
        }),
      t.organization.created,
    );
  };

  const handleJoinOrg = async () => {
    if (!joinOrgId.trim()) {
      showToast(t.organization.idRequired, "error");
      return;
    }
    await runAction(
      () => api.joinOrganization({ organization_id: joinOrgId.trim() }),
      t.organization.joinRequested,
    );
  };

  const handleInvite = async () => {
    if (!invitee.trim()) {
      showToast(t.organization.inviteeRequired, "error");
      return;
    }
    setInviteSubmitting(true);
    try {
      const response = await api.inviteOrganizationMember(invitee.trim());
      setInvitee("");
      showToast(
        `${t.organization.inviteCreated}: ${response.invitee}`,
        "success",
      );
    } catch (error) {
      showToast(`${t.status.error}: ${error}`, "error");
    } finally {
      setInviteSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 normal-case">
      <Toast toast={toast} />

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card className="overflow-hidden rounded-lg border-border bg-card">
          <CardContent className="grid gap-6 p-6 md:grid-cols-[1.15fr_0.85fr] md:p-7">
            <div className="grid gap-5">
              <div className="grid gap-3">
                <div className="inline-flex w-fit items-center gap-2 rounded-full border border-border bg-muted/40 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                  <Network className="h-3.5 w-3.5 text-primary" />
                  {t.organization.eyebrow}
                </div>
                <div className="grid gap-2">
                  <h1 className="text-3xl font-semibold tracking-[-0.03em] text-foreground md:text-4xl">
                    {t.organization.title}
                  </h1>
                  <p className="max-w-xl text-sm leading-6 text-muted-foreground">
                    {t.organization.subtitle}
                  </p>
                </div>
              </div>

              <div className="grid gap-3 rounded-lg border border-border bg-muted/25 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="grid gap-1">
                    <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                      {t.organization.currentOrg}
                    </span>
                    <div className="text-xl font-semibold tracking-[-0.02em] text-foreground">
                      {current?.organization_name ?? t.organization.personalOnlyMode}
                    </div>
                  </div>
                  <Badge tone={statusTone(current?.membership_status ?? "unassigned")}>
                    {t.organization.statusLabel}: {statusLabel(current?.membership_status ?? "unassigned")}
                  </Badge>
                </div>

                <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                  {currentOrgId ? (
                    <Badge tone="secondary">{currentOrgId}</Badge>
                  ) : (
                    <Badge tone="outline">{t.organization.noneAssigned}</Badge>
                  )}
                  <span>
                    {t.organization.roleLabel}: {current?.member_role ?? t.common.none}
                  </span>
                </div>

                <div className="flex items-center justify-between gap-4 rounded-md border border-border bg-background/70 px-4 py-3">
                  <div className="grid gap-1">
                    <div className="text-sm font-medium text-foreground">
                      {t.organization.sharingTitle}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {t.organization.sharingDescription}
                    </div>
                  </div>
                  <Switch
                    checked={Boolean(current?.sharing_enabled)}
                    disabled={
                      submitting ||
                      current?.membership_status !== "active" ||
                      !currentOrgId
                    }
                    onCheckedChange={(checked: boolean) =>
                      runAction(
                        () => api.setOrganizationSharing(checked),
                        checked ? t.organization.sharingEnabled : t.organization.sharingDisabled,
                      )
                    }
                  />
                </div>
              </div>

              <div className="flex flex-wrap gap-3">
                <Button
                  onClick={load}
                  outlined
                  className="cursor-pointer gap-2 rounded-[var(--radius-md)] border-border bg-background text-foreground hover:bg-muted"
                  disabled={submitting || inviteSubmitting}
                >
                  <RefreshCw className="h-4 w-4" />
                  {t.common.refresh}
                </Button>
              </div>
            </div>

            <div className="grid gap-3 rounded-lg border border-border bg-background/65 p-4">
              <div className="grid gap-1">
                <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                  {t.organization.membershipsTitle}
                </div>
                <div className="text-sm text-muted-foreground">
                  {t.organization.joinHint}
                </div>
              </div>

              {memberships.length === 0 ? (
                <div className="rounded-md border border-dashed border-border px-4 py-5 text-sm text-muted-foreground">
                  {t.organization.noMemberships}
                </div>
              ) : (
                memberships.map((membership) => {
                  const active = membership.organization_id === currentOrgId;
                  return (
                    <div
                      key={membership.organization_id}
                      className={cn(
                        "grid gap-3 rounded-md border px-4 py-3",
                        active
                          ? "border-primary/35 bg-primary/10"
                          : "border-border bg-background/55",
                      )}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="grid gap-1">
                          <div className="font-medium text-foreground">
                            {membership.organization_name || membership.organization_id}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {membership.organization_id}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Badge tone={statusTone(membership.membership_status)}>
                            {statusLabel(membership.membership_status)}
                          </Badge>
                          {active && <Badge tone="secondary">{t.organization.current}</Badge>}
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-3 text-sm text-muted-foreground">
                        <span>
                          {t.organization.roleLabel}: {membership.member_role ?? t.common.none}
                        </span>
                        <Button
                          outlined
                          size="sm"
                          disabled={submitting || active}
                          className="cursor-pointer rounded-[var(--radius-md)] border-border bg-background hover:bg-muted"
                          onClick={() =>
                            runAction(
                              () => api.switchOrganization(membership.organization_id),
                              t.organization.switched,
                            )
                          }
                        >
                          {t.organization.switchAction}
                        </Button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-lg border-border bg-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Users className="h-4 w-4 text-primary" />
              {t.organization.currentAccess}
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-5">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="grid gap-2 rounded-md border border-border bg-background/55 p-4">
                <Label htmlFor="organization-name">{t.organization.createOrganization}</Label>
                <Input
                  id="organization-name"
                  value={orgName}
                  onChange={(event) => setOrgName(event.target.value)}
                  placeholder={t.organization.namePlaceholder}
                />
                <Input
                  value={createOrgId}
                  onChange={(event) => setCreateOrgId(event.target.value)}
                  placeholder={t.organization.idPlaceholder}
                />
                <Button
                  onClick={handleCreateOrg}
                  disabled={submitting}
                  className="cursor-pointer gap-2 rounded-[var(--radius-md)] bg-primary text-primary-foreground transition-transform hover:scale-[1.05] hover:bg-primary/90 active:scale-[0.95]"
                >
                  <Users className="h-4 w-4" />
                  {t.organization.createOrganization}
                </Button>
              </div>
              <div className="grid gap-2 rounded-md border border-border bg-background/55 p-4">
                <Label htmlFor="join-organization">{t.organization.joinOrganization}</Label>
                <Input
                  id="join-organization"
                  value={joinOrgId}
                  onChange={(event) => setJoinOrgId(event.target.value)}
                  placeholder={t.organization.idPlaceholder}
                />
                <p className="text-sm text-muted-foreground">
                  {t.organization.joinHint}
                </p>
                <Button
                  onClick={handleJoinOrg}
                  disabled={submitting}
                  outlined
                  className="cursor-pointer gap-2 rounded-[var(--radius-md)] border-border bg-background hover:bg-muted"
                >
                  <ArrowRightLeft className="h-4 w-4" />
                  {t.organization.joinOrganization}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {(current?.can_invite_members || members.length > 0) && (
        <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
          <Card className="rounded-lg border-border bg-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Users className="h-4 w-4 text-primary" />
                {t.organization.membersTitle}
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-3">
                <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                  {t.organization.activeMembers.replace("{count}", String(activeMembers.length))}
                </div>
                {activeMembers.map((member) => (
                  <div
                    key={member.user_id}
                    className="flex items-center justify-between gap-3 rounded-md border border-border bg-background/55 p-4"
                  >
                    <div>
                      <div className="font-medium text-foreground">{member.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {member.member_role ?? t.common.none} · {member.workspace_slug}
                      </div>
                    </div>
                    <Badge tone="secondary">
                      {member.sharing_enabled ? t.common.enabled : t.common.disabled}
                    </Badge>
                  </div>
                ))}
              </div>

              {pendingMembers.length > 0 && (
                <div className="grid gap-3">
                  <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                    {t.organization.pendingMembers.replace("{count}", String(pendingMembers.length))}
                  </div>
                  {pendingMembers.map((member) => (
                    <div
                      key={member.user_id}
                      className="flex items-center justify-between gap-3 rounded-md border border-border bg-background/55 p-4"
                    >
                      <div>
                        <div className="font-medium text-foreground">{member.name}</div>
                        <div className="text-sm text-muted-foreground">{member.workspace_slug}</div>
                      </div>
                      {current?.can_change_settings ? (
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            disabled={submitting}
                            className="cursor-pointer rounded-[var(--radius-md)] bg-primary text-primary-foreground transition-transform hover:scale-[1.05] hover:bg-primary/90 active:scale-[0.95]"
                            onClick={() =>
                              runAction(
                                () => api.approveOrganizationMember(member.user_id),
                                t.organization.memberApproved,
                              )
                            }
                          >
                            <Check className="h-4 w-4" />
                          </Button>
                          <Button
                            size="sm"
                            outlined
                            disabled={submitting}
                            className="cursor-pointer rounded-[var(--radius-md)] border-border bg-background hover:bg-muted"
                            onClick={() =>
                              runAction(
                                () => api.removeOrganizationMember(member.user_id),
                                t.organization.memberRemoved,
                              )
                            }
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <Badge tone="warning">{statusLabel("pending")}</Badge>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="rounded-lg border-border bg-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <UserPlus className="h-4 w-4 text-primary" />
                {t.organization.auditAndInvites}
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4">
              {current?.can_invite_members && (
                <div className="grid gap-2 rounded-md border border-border bg-background/55 p-4">
                  <Label htmlFor="invitee">{t.organization.inviteTitle}</Label>
                  <Input
                    id="invitee"
                    value={invitee}
                    onChange={(event) => setInvitee(event.target.value)}
                    placeholder={t.organization.invitePlaceholder}
                  />
                  <Button
                    onClick={handleInvite}
                    disabled={inviteSubmitting}
                    className="cursor-pointer gap-2 rounded-[var(--radius-md)] bg-primary text-primary-foreground transition-transform hover:scale-[1.05] hover:bg-primary/90 active:scale-[0.95]"
                  >
                    <UserPlus className="h-4 w-4" />
                    {t.organization.inviteAction}
                  </Button>
                </div>
              )}

              <div className="grid gap-3">
                <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                  {t.organization.auditTitle}
                </div>
                {auditEvents.length === 0 ? (
                  <div className="rounded-md border border-dashed border-border px-4 py-6 text-sm text-muted-foreground">
                    {t.organization.noAuditEvents}
                  </div>
                ) : (
                  auditEvents
                    .slice()
                    .reverse()
                    .map((event) => (
                      <div
                        key={event.event_id}
                        className="rounded-md border border-border bg-background/55 p-4"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="font-medium text-foreground">
                            {event.event_type}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {event.created_at}
                          </div>
                        </div>
                        <div className="mt-2 text-sm text-muted-foreground">
                          {t.organization.actorLabel}: {event.actor_user_id}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {t.organization.subjectLabel}: {event.subject_user_id}
                        </div>
                      </div>
                    ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
