export const deUpdateChannel = {
  production: "Produktion",
  development: "Entwicklung",
  packagedDescription: "Produktion verwendet veröffentlichte Versionen. Entwicklung enthält Testversionen. Der Wechsel prüft nur auf Updates; die Installation bleibt Ihre Entscheidung.",
  title: 'Update-Kanal',
  description:
    'Stabil installiert nur veröffentlichte Releases. Jeder Commit folgt dem main-Branch, sobald Änderungen eintreffen.',
  stable: 'Stabile Releases',
  main: 'Jeder Commit',
  stableConfirmTitle: 'Zu stabilen Releases wechseln?',
  stableConfirmBody:
    'Semantier wechselt zum neuesten veröffentlichten Release, das älter sein kann als der Code, den du gerade nutzt. Von neuerem Code geschriebene Daten werden eventuell nicht korrekt gelesen – sichere sie vorher.',
  stableConfirm: 'Zu Stabil wechseln',
  failed: 'Update-Kanal konnte nicht geändert werden',
  detailsLabel: 'Kanal',
  branch: (b: string) => `Branch: ${b}`,
  change: 'Ändern',
  latestRelease: (v: string) => `Du nutzt das neueste stabile Release (${v}).`,
  rollbackTitle: (v: string) => `Zurück zum stabilen Release ${v}`,
  rollbackBody: (v: string) =>
    `Dieser Build ist neuer als das stabile Release ${v}. Die Installation setzt Semantier auf dieses Release zurück.`,
  rollbackAction: (v: string) => `Zurück auf ${v}`
}
