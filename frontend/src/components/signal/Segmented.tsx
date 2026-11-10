import "./signal.css";

export interface SegmentedOption {
  value: string;
  label: string;
}

interface SegmentedProps {
  /** Group label; required — a segmented control with no name is unreadable. */
  ariaLabel: string;
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
}

/**
 * Single-choice control for 2–5 short labels (spec §3.4).
 *
 * A `role="group"` of toggle buttons rather than a `<select>`: the options stay
 * visible, and it scrolls horizontally instead of wrapping into a second line
 * at 375px.
 */
export default function Segmented({ ariaLabel, options, value, onChange }: SegmentedProps) {
  return (
    <div className="sig-seg" role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="sig-seg-item"
          aria-pressed={option.value === value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
