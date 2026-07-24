# Miner submission terms

> **Status: draft.** These terms have not been reviewed by counsel and are not
> in effect. They become binding only when a subnet announcement designates a
> specific version of this document as the terms of participation. Obtain legal
> review and bind the operator identity before activation.

These terms define the rights needed for the subnet operator to evaluate,
improve, integrate, and commercially operate submitted work, including as part
of a hosted inference service. Subnet emissions are the sole compensation under
this draft. The repository's
[Apache-2.0 license](https://github.com/latent-to/cacheon/blob/main/LICENSE) covers the
repository code; it does not automatically license source submitted by miners.

## 1. Acceptance

Submitting a bundle to the subnet by committing its content hash and fetch URL
through the hotkey-signed `optima chain-submit` path constitutes acceptance of
the version of these terms designated at the commitment block. The signed
on-chain commitment is the record of acceptance.

## 2. License grant to the operator

For each submission, you grant the subnet operator a **perpetual, irrevocable,
worldwide, non-exclusive, royalty-free, sublicensable, transferable license**
to use, reproduce, modify, adapt, create derivative works of, distribute,
publicly perform and display, and commercially exploit the submission and its
derivatives, alone or combined with other software. This grant includes use in
hosted inference services and other commercial offerings.

## 3. Compensation

Subnet emissions earned under the active scoring and incentive mechanism are
the **sole and complete compensation** for a submission and for every right
granted under these terms. No royalty, revenue share, or other payment is owed
for use of a submission, including commercial use in an inference service.

## 4. Your representations

By submitting, you represent that:

1. the submission is your original work, or you have sufficient rights to
   submit it and make these grants;
2. the submission includes third-party code only when its license permits the
   submission and grant, and you have complied with that license;
3. the submission contains no code intended to subvert evaluation, exfiltrate
   data, damage systems, or interfere with other participants; and
4. you have authority to bind the entity for which you submit, if any.

The validator's static policy, isolation, and audit controls are independent of
these representations. See the [threat model](../security/threat-model.md).

## 5. Rights retained by the submitter

You retain copyright in your submission. The grant above is a license, not an
assignment. These terms do not restrict your right to use, publish, or license
your own work elsewhere. Copy detection and settlement determine what earns on
the subnet; they do not prohibit off-subnet use of your own work.

## 6. Activation terms that remain unresolved

### Public licensing of revealed bundles

After reveal, a bundle's fetch URL is public chain data. Public visibility does
not by itself grant reuse rights. The operator must decide before activation
whether revealed submissions receive an additional public license or remain
all-rights-reserved except for the operator grant above.

### Operator identity

“Subnet operator” must be bound to a concrete legal entity before these terms
can take effect.

## 7. Changes

Terms may be updated prospectively. A submission is governed by the version
designated when its commitment was signed; later updates do not apply
retroactively.
