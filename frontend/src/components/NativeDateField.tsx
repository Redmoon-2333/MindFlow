import { localDateStr } from "../date-utils";

/** Browser calendar with a separate label and local-day validation. */
export default function NativeDateField({ label, value, onChange, allowEmpty = false }: {
  label: string; value: string; onChange: (value: string) => void; allowEmpty?: boolean;
}) {
  const today = localDateStr(new Date());
  return <label className="native-date-field"><span>{label}</span><input type="date" aria-label={label} value={value} max={today}
    onChange={event => {
      const next = event.currentTarget.value;
      if (!next && allowEmpty) onChange("");
      else if (next && next <= today && event.currentTarget.validity.valid) onChange(next);
    }} /></label>;
}
