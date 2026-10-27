import {
  AppWindow as AsimovAppWindow,
  Archive as AsimovArchive,
  ArrowClockwise as AsimovArrowClockwise,
  ArrowElbowDownLeft as AsimovArrowElbowDownLeft,
  ArrowLineDown as AsimovArrowLineDown,
  ArrowLineUp as AsimovArrowLineUp,
  ArrowSquareOut as AsimovArrowSquareOut,
  ArrowUp as AsimovArrowUp,
  ArrowUpRight as AsimovArrowUpRight,
  At as AsimovAt,
  Bell as AsimovBell,
  Book as AsimovBook,
  BookmarkSimple as AsimovBookmarkSimple,
  Brain as AsimovBrain,
  Browser as AsimovBrowser,
  Bug as AsimovBug,
  CaretDown as AsimovCaretDown,
  CaretLeft as AsimovCaretLeft,
  CaretRight as AsimovCaretRight,
  CaretUp as AsimovCaretUp,
  ChartBar as AsimovChartBar,
  ChatCircle as AsimovChatCircle,
  ChatCircleDots as AsimovChatCircleDots,
  ChatText as AsimovChatText,
  Check as AsimovCheck,
  CheckCircle as AsimovCheckCircle,
  Circle as AsimovCircle,
  Clipboard as AsimovClipboard,
  Clock as AsimovClock,
  Cloud as AsimovCloud,
  Command as AsimovCommand,
  Copy as AsimovCopy,
  CornersOut as AsimovCornersOut,
  Cpu as AsimovCpu,
  CreditCard as AsimovCreditCard,
  Cube as AsimovCube,
  DotsThree as AsimovDotsThree,
  DotsThreeVertical as AsimovDotsThreeVertical,
  DownloadSimple as AsimovDownloadSimple,
  Ear as AsimovEar,
  EarSlash as AsimovEarSlash,
  Egg as AsimovEgg,
  Eject as AsimovEject,
  Envelope as AsimovEnvelope,
  Eye as AsimovEye,
  EyeSlash as AsimovEyeSlash,
  FileImage as AsimovFileImage,
  FileText as AsimovFileText,
  FloppyDisk as AsimovFloppyDisk,
  FolderOpen as AsimovFolderOpen,
  Gear as AsimovGear,
  GearSix as AsimovGearSix,
  GitBranch as AsimovGitBranch,
  GitFork as AsimovGitFork,
  Globe as AsimovGlobe,
  Graph as AsimovGraph,
  Hash as AsimovHash,
  Image as AsimovImage,
  Info as AsimovInfo,
  Key as AsimovKey,
  Keyboard as AsimovKeyboard,
  Layout as AsimovLayout,
  Lightning as AsimovLightning,
  Link as AsimovLink,
  Lock as AsimovLock,
  MagnifyingGlass as AsimovMagnifyingGlass,
  MagnifyingGlassMinus as AsimovMagnifyingGlassMinus,
  MagnifyingGlassPlus as AsimovMagnifyingGlassPlus,
  Microphone as AsimovMicrophone,
  MicrophoneSlash as AsimovMicrophoneSlash,
  Monitor as AsimovMonitor,
  MonitorPlay as AsimovMonitorPlay,
  Moon as AsimovMoon,
  Notebook as AsimovNotebook,
  Package as AsimovPackage,
  Palette as AsimovPalette,
  PaperPlaneTilt as AsimovPaperPlaneTilt,
  Pause as AsimovPause,
  PawPrint as AsimovPawPrint,
  PencilLine as AsimovPencilLine,
  PencilSimple as AsimovPencilSimple,
  Play as AsimovPlay,
  Plug as AsimovPlug,
  Plus as AsimovPlus,
  Power as AsimovPower,
  Pulse as AsimovPulse,
  PushPin as AsimovPushPin,
  PuzzlePiece as AsimovPuzzlePiece,
  QrCode as AsimovQrCode,
  Question as AsimovQuestion,
  ShieldCheck as AsimovShieldCheck,
  SidebarSimple as AsimovSidebarSimple,
  SignIn as AsimovSignIn,
  SlidersHorizontal as AsimovSlidersHorizontal,
  Smiley as AsimovSmiley,
  SpeakerHigh as AsimovSpeakerHigh,
  SpeakerSlash as AsimovSpeakerSlash,
  SpinnerGap as AsimovSpinnerGap,
  Square as AsimovSquare,
  SquaresFour as AsimovSquaresFour,
  Stack as AsimovStack,
  SteeringWheel as AsimovSteeringWheel,
  Stop as AsimovStop,
  Sun as AsimovSun,
  TerminalWindow as AsimovTerminalWindow,
  TextAa as AsimovTextAa,
  Trash as AsimovTrash,
  UploadSimple as AsimovUploadSimple,
  Users as AsimovUsers,
  Warning as AsimovWarning,
  WarningCircle as AsimovWarningCircle,
  Waveform as AsimovWaveform,
  Wrench as AsimovWrench,
  X as AsimovX,
  type IconWeight,
  type Icon as PhosphorIcon
} from '@phosphor-icons/react'
import * as Legacy from '@tabler/icons-react'
import type { Icon as IconComponent } from '@tabler/icons-react'
import { createContext, createElement, useContext } from 'react'

/** ThemeProvider owns the visual grammar; icons retain existing names/props and
 * native/brand glyphs remain independent of this UI icon boundary. */
export const AsimovIconContext = createContext(false)

function themedIcon(legacy: IconComponent, modern: PhosphorIcon, weight: IconWeight = 'thin'): IconComponent {
  return function ThemedIcon(props: Legacy.IconProps) {
    const asimov = useContext(AsimovIconContext)

    return asimov
      ? createElement(modern, { ...props, stroke: undefined, size: props.size ?? 24, weight })
      : createElement(legacy, props)
  }
}

export const Activity = themedIcon(Legacy.IconActivity, AsimovPulse)
export const AlertCircle = themedIcon(Legacy.IconAlertCircle, AsimovWarningCircle)
export const AlertTriangle = themedIcon(Legacy.IconAlertTriangle, AsimovWarning)
export const AppWindow = themedIcon(Legacy.IconAppWindow, AsimovAppWindow)
export const Archive = themedIcon(Legacy.IconArchive, AsimovArchive)
export const ArchiveOff = themedIcon(Legacy.IconArchiveOff, AsimovArchive)
export const ArrowBarToDown = themedIcon(Legacy.IconArrowBarToDown, AsimovArrowLineDown)
export const ArrowBarToUp = themedIcon(Legacy.IconArrowBarToUp, AsimovArrowLineUp)
export const ArrowUp = themedIcon(Legacy.IconArrowUp, AsimovArrowUp)
export const ArrowUpRight = themedIcon(Legacy.IconArrowUpRight, AsimovArrowUpRight)
export const AtSign = themedIcon(Legacy.IconAt, AsimovAt)
export const AudioLines = themedIcon(Legacy.IconWaveSine, AsimovWaveform)
export const BarChart3 = themedIcon(Legacy.IconChartBar, AsimovChartBar)
export const Bell = themedIcon(Legacy.IconBell, AsimovBell)
export const Book = themedIcon(Legacy.IconBook, AsimovBook)
export const Bookmark = themedIcon(Legacy.IconBookmark, AsimovBookmarkSimple)
export const BookmarkFilled = themedIcon(Legacy.IconBookmarkFilled, AsimovBookmarkSimple, 'fill')
export const Box = themedIcon(Legacy.IconBox, AsimovCube)
export const Brain = themedIcon(Legacy.IconBrain, AsimovBrain)
export const Bug = themedIcon(Legacy.IconBug, AsimovBug)
export const Check = themedIcon(Legacy.IconCheck, AsimovCheck)
export const CheckCircle2 = themedIcon(Legacy.IconCircleCheck, AsimovCheckCircle)
export const CheckIcon = themedIcon(Legacy.IconCheck, AsimovCheck)
export const ChevronDown = themedIcon(Legacy.IconChevronDown, AsimovCaretDown)
export const ChevronDownIcon = themedIcon(Legacy.IconChevronDown, AsimovCaretDown)
export const ChevronLeft = themedIcon(Legacy.IconChevronLeft, AsimovCaretLeft)
export const ChevronLeftIcon = themedIcon(Legacy.IconChevronLeft, AsimovCaretLeft)
export const ChevronRight = themedIcon(Legacy.IconChevronRight, AsimovCaretRight)
export const ChevronRightIcon = themedIcon(Legacy.IconChevronRight, AsimovCaretRight)
export const ChevronUp = themedIcon(Legacy.IconChevronUp, AsimovCaretUp)
export const CircleIcon = themedIcon(Legacy.IconCircle, AsimovCircle)
export const CircleLetterA = themedIcon(Legacy.IconCircleLetterA, AsimovTextAa)
export const Clipboard = themedIcon(Legacy.IconClipboard, AsimovClipboard)
export const Clock = themedIcon(Legacy.IconClock, AsimovClock)
export const Cloud = themedIcon(Legacy.IconCloud, AsimovCloud)
export const Command = themedIcon(Legacy.IconCommand, AsimovCommand)
export const Copy = themedIcon(Legacy.IconCopy, AsimovCopy)
export const CopyIcon = themedIcon(Legacy.IconCopy, AsimovCopy)
export const CornerDownLeft = themedIcon(Legacy.IconCornerDownLeft, AsimovArrowElbowDownLeft)
export const Cpu = themedIcon(Legacy.IconCpu, AsimovCpu)
export const CreditCard = themedIcon(Legacy.IconCreditCard, AsimovCreditCard)
export const Download = themedIcon(Legacy.IconDownload, AsimovDownloadSimple)
export const Ear = themedIcon(Legacy.IconEar, AsimovEar)
export const EarOff = themedIcon(Legacy.IconEarOff, AsimovEarSlash)
export const Egg = themedIcon(Legacy.IconEgg, AsimovEgg)
export const Eject = themedIcon(Legacy.IconPlayerEjectFilled, AsimovEject, 'fill')
export const ExternalLink = themedIcon(Legacy.IconExternalLink, AsimovArrowSquareOut)
export const Eye = themedIcon(Legacy.IconEye, AsimovEye)
export const EyeOff = themedIcon(Legacy.IconEyeOff, AsimovEyeSlash)
export const FileImage = themedIcon(Legacy.IconPhoto, AsimovFileImage)
export const FileText = themedIcon(Legacy.IconFileText, AsimovFileText)
export const FolderOpen = themedIcon(Legacy.IconFolderOpen, AsimovFolderOpen)
export const GitBranch = themedIcon(Legacy.IconGitBranch, AsimovGitBranch)
export const GitBranchIcon = themedIcon(Legacy.IconGitBranch, AsimovGitBranch)
export const GitFork = themedIcon(Legacy.IconGitFork, AsimovGitFork)
export const GitForkIcon = themedIcon(Legacy.IconGitFork, AsimovGitFork)
export const Globe = themedIcon(Legacy.IconGlobe, AsimovGlobe)
export const Hash = themedIcon(Legacy.IconHash, AsimovHash)
export const HelpCircle = themedIcon(Legacy.IconHelpCircle, AsimovQuestion)
export const ImageIcon = themedIcon(Legacy.IconPhoto, AsimovImage)
export const Info = themedIcon(Legacy.IconInfoCircle, AsimovInfo)
export const Keyboard = themedIcon(Legacy.IconKeyboard, AsimovKeyboard)
export const KeyRound = themedIcon(Legacy.IconKey, AsimovKey)
export const Layers3 = themedIcon(Legacy.IconLayersIntersect2, AsimovStack)
export const LayoutDashboard = themedIcon(Legacy.IconLayoutDashboard, AsimovSquaresFour)
export const Link = themedIcon(Legacy.IconLink, AsimovLink)
export const Link2 = themedIcon(Legacy.IconLink, AsimovLink)
export const LinkIcon = themedIcon(Legacy.IconLink, AsimovLink)
export const Loader2 = themedIcon(Legacy.IconLoader2, AsimovSpinnerGap)
export const Loader2Icon = themedIcon(Legacy.IconLoader2, AsimovSpinnerGap)
export const Lock = themedIcon(Legacy.IconLock, AsimovLock)
export const LogIn = themedIcon(Legacy.IconLogin, AsimovSignIn)
export const Mail = themedIcon(Legacy.IconMail, AsimovEnvelope)
export const Maximize = themedIcon(Legacy.IconMaximize, AsimovCornersOut)
export const MessageCircle = themedIcon(Legacy.IconMessageCircle, AsimovChatCircle)
export const MessageCode = themedIcon(Legacy.IconMessageCode, AsimovChatText)
export const MessageQuestion = themedIcon(Legacy.IconMessageQuestion, AsimovChatCircleDots)
export const MessageSquareText = themedIcon(Legacy.IconMessage2, AsimovChatText)
export const Mic = themedIcon(Legacy.IconMicrophone, AsimovMicrophone)
export const MicOff = themedIcon(Legacy.IconMicrophoneOff, AsimovMicrophoneSlash)
export const Monitor = themedIcon(Legacy.IconDeviceDesktop, AsimovMonitor)
export const MonitorPlay = themedIcon(Legacy.IconDeviceDesktopAnalytics, AsimovMonitorPlay)
export const Moon = themedIcon(Legacy.IconMoon, AsimovMoon)
export const MoreHorizontal = themedIcon(Legacy.IconDots, AsimovDotsThree)
export const MoreHorizontalIcon = themedIcon(Legacy.IconDots, AsimovDotsThree)
export const MoreVertical = themedIcon(Legacy.IconDotsVertical, AsimovDotsThreeVertical)
export const Network = themedIcon(Legacy.IconNetwork, AsimovGraph)
export const NotebookTabs = themedIcon(Legacy.IconNotebook, AsimovNotebook)
export const Package = themedIcon(Legacy.IconPackage, AsimovPackage)
export const Palette = themedIcon(Legacy.IconPalette, AsimovPalette)
export const PanelBottom = themedIcon(Legacy.IconLayoutBottombar, AsimovLayout)
export const PanelLeftIcon = themedIcon(Legacy.IconLayoutSidebar, AsimovSidebarSimple)
export const PanelTop = themedIcon(Legacy.IconLayoutNavbar, AsimovBrowser)
export const Pause = themedIcon(Legacy.IconPlayerPause, AsimovPause)
export const PawPrint = themedIcon(Legacy.IconPaw, AsimovPawPrint)
export const Pencil = themedIcon(Legacy.IconPencil, AsimovPencilSimple)
export const PencilIcon = themedIcon(Legacy.IconPencil, AsimovPencilSimple)
export const PencilLine = themedIcon(Legacy.IconPencil, AsimovPencilLine)
export const Pin = themedIcon(Legacy.IconPin, AsimovPushPin)
export const Play = themedIcon(Legacy.IconPlayerPlay, AsimovPlay)
export const Plug = themedIcon(Legacy.IconPlug, AsimovPlug)
export const Plus = themedIcon(Legacy.IconPlus, AsimovPlus)
export const Power = themedIcon(Legacy.IconPower, AsimovPower)
export const QrCode = themedIcon(Legacy.IconQrcode, AsimovQrCode)
export const RefreshCw = themedIcon(Legacy.IconRefresh, AsimovArrowClockwise)
export const RefreshCwIcon = themedIcon(Legacy.IconRefresh, AsimovArrowClockwise)
export const Save = themedIcon(Legacy.IconDeviceFloppy, AsimovFloppyDisk)
export const Search = themedIcon(Legacy.IconSearch, AsimovMagnifyingGlass)
export const SearchIcon = themedIcon(Legacy.IconSearch, AsimovMagnifyingGlass)
export const Send = themedIcon(Legacy.IconSend, AsimovPaperPlaneTilt)
export const Settings = themedIcon(Legacy.IconSettings, AsimovGear)
export const Settings2 = themedIcon(Legacy.IconSettings2, AsimovGearSix)
export const ShieldLock = themedIcon(Legacy.IconShieldLock, AsimovShieldCheck)
export const SlidersHorizontal = themedIcon(Legacy.IconAdjustmentsHorizontal, AsimovSlidersHorizontal)
export const SmilePlusIcon = themedIcon(Legacy.IconMoodPlus, AsimovSmiley)
export const Square = themedIcon(Legacy.IconSquare, AsimovSquare)
export const Starmap = themedIcon(Legacy.IconChartDots3, AsimovGraph)
export const SteeringWheel = themedIcon(Legacy.IconSteeringWheel, AsimovSteeringWheel)
export const StopFilled = themedIcon(Legacy.IconPlayerStopFilled, AsimovStop, 'fill')
export const Sun = themedIcon(Legacy.IconSun, AsimovSun)
export const Terminal = themedIcon(Legacy.IconTerminal2, AsimovTerminalWindow)
export const Trash2 = themedIcon(Legacy.IconTrash, AsimovTrash)
export const Upload = themedIcon(Legacy.IconUpload, AsimovUploadSimple)
export const Users = themedIcon(Legacy.IconUsers, AsimovUsers)
export const Volume2 = themedIcon(Legacy.IconVolume2, AsimovSpeakerHigh)
export const Volume2Icon = themedIcon(Legacy.IconVolume2, AsimovSpeakerHigh)
export const VolumeX = themedIcon(Legacy.IconVolumeOff, AsimovSpeakerSlash)
export const VolumeXIcon = themedIcon(Legacy.IconVolumeOff, AsimovSpeakerSlash)
export const Wrench = themedIcon(Legacy.IconTool, AsimovWrench)
export const X = themedIcon(Legacy.IconX, AsimovX)
export const XIcon = themedIcon(Legacy.IconX, AsimovX)
export const Zap = themedIcon(Legacy.IconBolt, AsimovLightning)
export const ZapFilled = themedIcon(Legacy.IconBoltFilled, AsimovLightning, 'fill')
export const ZoomIn = themedIcon(Legacy.IconZoomIn, AsimovMagnifyingGlassPlus)
export const ZoomOut = themedIcon(Legacy.IconZoomOut, AsimovMagnifyingGlassMinus)
export const Puzzle = themedIcon(Legacy.IconPuzzle, AsimovPuzzlePiece)

export type { IconComponent }

/**
 * Named icon-size scale — the single source of truth for SVG icon dimensions,
 * replacing ad-hoc `h-N w-N` / `size={N}`. Use `<Icon className={iconSize.sm} />`
 * (Tailwind `size-*` sets w+h and beats the icon's default 24px); compose with
 * `cn()` for colour/animation classes.
 */
export const iconSize = {
  xs: 'size-3', // 12px
  sm: 'size-3.5', // 14px
  md: 'size-4', // 16px
  lg: 'size-5', // 20px
  xl: 'size-6' // 24px
} as const

export type IconSize = keyof typeof iconSize
