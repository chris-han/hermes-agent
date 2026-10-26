import type { FirstRunCopy } from './types_first_run'

export const frFirstRun: FirstRunCopy = {
  beats: {
    name: 'Nom',
    theme: "Couleur d'accent",
    layout: 'Disposition',
    connectors: 'Applications',
    plugins: 'Plugins',
    tour: 'Visite',
    build: 'Première tâche'
  },
  namePlaceholder: 'Votre nom',
  useName: name => `Utiliser « ${name} »`,
  install: 'Installer',
  installed: 'Installé',
  loading: 'Chargement…',
  noPlugins: 'Aucun plugin à afficher pour le moment.',
  noneChosen: 'Aucun',
  noAnswer: 'Pas de réponse',
  answerRefused: 'Réponse non acceptée',
  tours: {
    quick: 'Visite rapide',
    full: 'Montre-moi tout',
    skip: 'Passer, construisons quelque chose'
  },
  skippedAll: 'Configuration ignorée',
  skip: 'Ignorer la configuration',
  skipTip: 'Termine la configuration guidée et ouvre une nouvelle discussion.',
  skipFailed: 'Impossible d’ignorer la configuration',
  intro: {
    greeting: 'Bonjour, je suis Semantier.',
    plan: 'Configurons tout pour vous. Ensuite, nous ferons quelque chose de génial.'
  },
  rerun: {
    title: "Relancer l'accueil",
    description: 'Ouvre un nouveau chat de configuration qui reprend les premières étapes avec votre modèle actuel.',
    action: 'Relancer',
    failed: "Impossible de lancer l'accueil"
  }
}
